"""LLM local vía Ollama (http://127.0.0.1:11434). Todo es opcional: si Ollama no está, se devuelven fallbacks."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

OLLAMA = "http://127.0.0.1:11434"


def _post(path: str, body: dict, timeout: float = 120):
    req = urllib.request.Request(OLLAMA + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


_start_tried = False


def _find_exe():
    import shutil
    cands = [os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
             os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"), shutil.which("ollama")]
    return next((c for c in cands if c and os.path.isfile(c)), None)


def _try_start():
    """Si Ollama está instalado pero no arrancado (y no está en el PATH), lo arranca en segundo plano."""
    global _start_tried
    if _start_tried:
        return False
    _start_tried = True
    exe = _find_exe()
    if not exe:
        return False
    import subprocess
    import time
    try:
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return False
    for _ in range(20):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(OLLAMA + "/api/tags", timeout=1).close()
            return True
        except Exception:
            pass
    return False


_tags_cache = None


def _tags() -> list[dict]:
    global _tags_cache
    for attempt in range(2):
        try:
            with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=1.5) as r:
                _tags_cache = json.load(r).get("models", [])
                return _tags_cache
        except Exception:
            if attempt or not _try_start():
                return []
    return []


def models() -> list[str]:
    return [m["name"] for m in _tags()]


def _size_b(m: dict) -> float:
    """Tamaño en miles de millones de parámetros (de los detalles de Ollama o del nombre)."""
    ps = str((m.get("details") or {}).get("parameter_size", ""))
    for src in (ps, m.get("name", "")):
        x = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", src)
        if x:
            return float(x.group(1))
    return 7.0 if m.get("size", 0) < 6e9 else 13.0


# Preferencias (de mejor a peor). Se evitan los gigantes (lentos y sin VRAM para la IA de vídeo),
# los de código y los de «razonamiento» (piensan minutos antes de responder).
TEXT_ORDER = ("qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b", "qwen3:8b", "gemma3:12b", "gemma2:9b", "qwen2.5",
              "llama3.1", "gemma3", "gemma2", "mistral", "llama3.2", "llama3", "qwen3", "phi")
VISION_ORDER = ("qwen2.5vl", "qwen2.5-vl", "llava:13b", "llava:7b", "llava", "llama3.2-vision", "minicpm-v",
                "gemma3", "bakllava", "moondream")
BAD_TEXT = ("coder", "code", "deepseek-r1", "r1", "embed", "vision", "llava", "moondream", "minicpm-v", "qwq")


def _pick(order, cands):
    for pref in order:
        for m in cands:
            n = m["name"].lower()
            if n == pref or n.startswith(pref + ":") or (":" in pref and n.startswith(pref)) or n.startswith(pref):
                return m["name"]
    return None


def text_model() -> str | None:
    env = os.environ.get("PIXELANIM_LLM")
    if env:
        return env
    ms = [m for m in _tags() if not any(b in m["name"].lower() for b in BAD_TEXT)]
    ok = [m for m in ms if 1 <= _size_b(m) <= 16]
    return _pick(TEXT_ORDER, ok) or (min(ok, key=lambda m: abs(_size_b(m) - 8))["name"] if ok else None)


def vision_model() -> str | None:
    env = os.environ.get("PIXELANIM_VISION")
    if env:
        return env
    ms = [m for m in _tags() if _size_b(m) <= 16]
    return _pick(VISION_ORDER, ms)


def _clean(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


# keep_alive=0: Ollama descarga el modelo de la VRAM al terminar, para dejar sitio a la IA de vídeo.
KEEP = os.environ.get("PIXELANIM_OLLAMA_KEEP", "0")
OPTS = {"num_ctx": 4096, "temperature": 0.4}


def chat_json(messages: list[dict], model: str | None = None, keep_alive=None) -> dict | None:
    model = model or text_model()
    if not model:
        return None
    try:
        out = _post("/api/chat", {"model": model, "messages": messages, "stream": False, "format": "json",
                                  "keep_alive": KEEP if keep_alive is None else keep_alive, "options": OPTS}, 90)
        txt = _clean(out["message"]["content"])
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0) if m else txt)
    except Exception:
        return None


def enhance_prompt(text: str) -> str:
    model = text_model()
    if not model:
        return text
    try:
        out = _post("/api/generate", {"model": model, "stream": False, "keep_alive": KEEP, "options": OPTS, "prompt": (
            "Rewrite this as a short English Stable Diffusion subject description for a looping animation "
            "(translate literally, keep the exact same subject; one subject, describe the motion, max 20 words, "
            "no style words, output only the text): " + text)}, 60)
        lines = [l for l in _clean(out["response"]).strip().strip('"').splitlines() if l.strip()]
        return lines[-1] if lines else text
    except Exception:
        return text


def describe_image(img) -> str | None:
    """Descripción corta en inglés del sujeto principal de una imagen (modelo de visión de Ollama)."""
    model = vision_model()
    if not model:
        return None
    b = io.BytesIO()
    im = img.convert("RGB")
    im.thumbnail((512, 512))
    im.save(b, "JPEG", quality=90)
    data = base64.b64encode(b.getvalue()).decode()
    prompts = ["What is the main subject of this image? Answer in under 12 English words.",
               "Describe this image briefly."]   # modelos pequeños (moondream) responden mejor a lo segundo
    for pr in prompts:
        try:
            out = _post("/api/generate", {"model": model, "stream": False, "keep_alive": KEEP, "images": [data],
                                          "prompt": pr, "options": {"num_ctx": 4096}}, 120)
            txt = _clean(out.get("response", "")).strip().strip('"')
            if txt:
                first = re.split(r"(?<=[.!?])\s", txt)[0]
                words = first.split()
                return " ".join(words[:16]).rstrip(".,;:")
        except Exception:
            continue
    return None
