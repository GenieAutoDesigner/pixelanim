"""Agente buscador de referencias: busca imágenes en internet, las descarga, las puntúa y deja las mejores.

Fuentes (sin cuentas ni claves):
  - DuckDuckGo Imágenes (indexa toda la web, incluidos Pinterest, Reddit, DeviantArt, ArtStation, Tumblr…)
  - Bing Imágenes (HTML público)
  - Openverse (Flickr y repositorios con licencia Creative Commons)
  - Wikimedia Commons
Las redes sociales con inicio de sesión obligatorio (Instagram, TikTok, X) no se raspan directamente: llegan sus
imágenes públicas indexadas por los buscadores, sobre todo con las consultas «site:».

Pasos:
  1. consultas: el LLM local (Ollama) las escribe en inglés; si no hay, se usan plantillas
  2. búsqueda en paralelo en todas las fuentes, con límite de tiempo por fuente
  3. descarga en paralelo (tiempo y tamaño máximos, se verifica que sea imagen, se eliminan duplicados por hash)
  4. puntuación: relevancia con CLIP (texto↔imagen, local) + «pixelabilidad» (fondo liso, contraste, sujeto centrado)
  5. opcional: recorte del sujeto con rembg sobre fondo blanco
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import html
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 PixelAnim/1.0")
MAX_BYTES = 12 * 2**20
TIMEOUT = 10


def _log(cb, msg):
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def _get(url: str, timeout: float = TIMEOUT, max_bytes: int = MAX_BYTES, accept: str = "*/*", ua: str = UA) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": accept, "Accept-Language": "en,es;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = int(r.headers.get("Content-Length") or 0)
        if n > max_bytes:
            raise ValueError("demasiado grande")
        data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("demasiado grande")
        return data


# ------------------------------------------------------------------ consultas

def make_queries(text: str, n: int = 4, use_llm: bool = True) -> list[str]:
    subject = text.strip()
    if use_llm:
        import llm
        out = llm.chat_json([
            {"role": "system", "content": "You write image search queries. Answer only JSON: {\"subject\": str, \"queries\": [str]}"},
            {"role": "user", "content": (
                f"User request (any language): {text}\n"
                "First translate the subject into English LITERALLY and precisely (keep the exact animal/object: "
                "búho=owl, lechuza=barn owl, zorro=fox, murciélago=bat, cuervo=raven; never replace it with a "
                "similar one).\n"
                f"Give the main subject in English (max 6 words) and {n} English web image search queries to find "
                "reference pictures of that subject that are easy to turn into a black silhouette: one plain "
                "'X silhouette', one 'X black and white illustration', one 'X isolated white background', and one "
                "with an action/pose. Short queries, no quotes.")}], keep_alive=0)
        if isinstance(out, dict) and out.get("queries"):
            qs = [str(q).strip() for q in out["queries"] if str(q).strip()][:n]
            subject = str(out.get("subject") or subject)
            if qs:
                return [subject] + qs
    return [subject, f"{subject} silhouette", f"{subject} black and white illustration",
            f"{subject} isolated on white background", f"{subject} pixel art"][: n + 1]


# ------------------------------------------------------------------ fuentes

def src_ddg(q: str, k: int):
    from ddgs import DDGS
    out = []
    for r in DDGS(timeout=TIMEOUT).images(q, max_results=k, safesearch="moderate"):
        out.append({"url": r.get("image"), "thumb": r.get("thumbnail"), "page": r.get("url"), "title": r.get("title", ""),
                    "w": r.get("width"), "h": r.get("height"), "source": "duckduckgo"})
    return out


def src_bing(q: str, k: int):
    page = _get("https://www.bing.com/images/search?" + urllib.parse.urlencode({"q": q, "form": "HDRSC2", "first": 1}),
                accept="text/html").decode("utf-8", "ignore")
    out = []
    for m in re.finditer(r'm="(\{.*?\})"', page):
        try:
            d = json.loads(html.unescape(m.group(1)))
        except Exception:
            continue
        if d.get("murl"):
            out.append({"url": d["murl"], "thumb": d.get("turl"), "page": d.get("purl"), "title": d.get("t", ""),
                        "source": "bing"})
        if len(out) >= k:
            break
    return out


def src_openverse(q: str, k: int):
    data = json.loads(_get("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(
        {"q": q, "page_size": min(k, 20), "mature": "false"}), accept="application/json"))
    return [{"url": r.get("url"), "thumb": r.get("thumbnail"), "page": r.get("foreign_landing_url"),
             "title": r.get("title", ""), "license": f"{r.get('license', '')} {r.get('license_version', '')}".strip(),
             "w": r.get("width"), "h": r.get("height"), "source": "openverse"} for r in data.get("results", [])]


def src_commons(q: str, k: int):
    data = json.loads(_get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {q}",
        "gsrnamespace": 6, "gsrlimit": min(k, 20), "prop": "imageinfo", "iiprop": "url|size", "iiurlwidth": 800}),
        accept="application/json", ua="PixelAnim/1.0 (local personal tool; https://github.com/) python-urllib"))
    out = []
    for p in (data.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("thumburl") or ii.get("url"):
            out.append({"url": ii.get("thumburl") or ii.get("url"), "page": ii.get("descriptionurl"),
                        "title": p.get("title", ""), "source": "commons"})
    return out


SOURCES = {"duckduckgo": src_ddg, "bing": src_bing, "openverse": src_openverse, "commons": src_commons}
SOCIAL_SITES = ["pinterest.com", "deviantart.com", "reddit.com", "artstation.com"]


def search(queries: list[str], per_query: int = 12, sources=None, social: bool = True, log=None) -> list[dict]:
    sources = sources or list(SOURCES)
    jobs = [(s, q) for q in queries for s in sources]
    if social and "duckduckgo" in sources:
        base = queries[1] if len(queries) > 1 else queries[0]
        jobs += [("duckduckgo", f"{base} site:{site}") for site in SOCIAL_SITES]
    found, seen = [], set()
    # una cola por fuente (secuencial dentro de cada fuente para no provocar bloqueos por exceso de peticiones),
    # todas las fuentes en paralelo
    import threading
    web = threading.Lock()   # ddgs puede usar Bing por debajo: comparten cola
    locks = {s: (web if s in ("duckduckgo", "bing") else threading.Lock()) for s in SOURCES}

    def run(s, q):
        with locks[s]:
            for attempt in range(2):
                try:
                    r = SOURCES[s](q, per_query)
                    time.sleep(1.0)
                    return r
                except Exception:
                    if attempt:
                        raise
                    time.sleep(2.5)

    with cf.ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        futs = {ex.submit(run, s, q): (s, q) for s, q in jobs}
        try:
            for f in cf.as_completed(futs, timeout=120):
                s, q = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    _log(log, f"  · {s} «{q}»: sin resultados ({type(e).__name__})")
                    continue
                n = 0
                for r in res:
                    u = r.get("url")
                    if u and u.startswith("http") and u not in seen:
                        seen.add(u)
                        r["query"] = q
                        found.append(r)
                        n += 1
                _log(log, f"  · {s} «{q}»: {n}")
        except cf.TimeoutError:
            _log(log, "  · algunas búsquedas tardaron demasiado y se ignoraron")
    return found


# ------------------------------------------------------------------ descarga

def _ahash(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((8, 8), Image.Resampling.BOX), np.float32)
    bits = (a > a.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def _fetch(item: dict):
    for u in (item.get("url"), item.get("thumb")):
        if not u:
            continue
        try:
            data = _get(u, accept="image/*")
            im = Image.open(io.BytesIO(data))
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
            if min(im.size) < 160:
                continue
            im.thumbnail((1024, 1024))
            return im
        except Exception:
            continue
    return None


def download(items: list[dict], max_n: int = 60, log=None) -> list[tuple[dict, Image.Image]]:
    items = items[: max_n]
    out, hashes = [], []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_fetch, it): it for it in items}
        try:
            for f in cf.as_completed(futs, timeout=90):
                im = f.result()
                if im is None:
                    continue
                h = _ahash(im)
                if any(bin(h ^ o).count("1") <= 4 for o in hashes):   # casi duplicada
                    continue
                hashes.append(h)
                out.append((futs[f], im))
        except cf.TimeoutError:
            _log(log, "  · descargas lentas ignoradas")
    _log(log, f"Descargadas {len(out)} imágenes válidas de {len(items)}")
    return out


# ------------------------------------------------------------------ puntuación

def pixelability(im: Image.Image) -> float:
    """0..1: fondo uniforme, buen contraste, sujeto de tamaño razonable y centrado."""
    a = np.asarray(im.convert("L").resize((96, 96), Image.Resampling.BOX), np.float32) / 255
    k = 6
    ring = np.concatenate([a[:k].ravel(), a[-k:].ravel(), a[:, :k].ravel(), a[:, -k:].ravel()])
    bg, bg_std = float(np.median(ring)), float(ring.std())
    diff = np.abs(a - bg)
    m = diff > 0.25
    cover = m.mean()
    if m.any():
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean() / 96, ys.mean() / 96
        center = 1 - min(1, ((cx - .5) ** 2 + (cy - .5) ** 2) ** .5 * 2)
    else:
        center = 0
    uniform = max(0.0, 1 - bg_std * 6)
    cov_s = max(0.0, 1 - abs(cover - 0.3) / 0.3)
    contrast = min(1.0, float(diff[m].mean()) * 1.6) if m.any() else 0
    return 0.4 * uniform + 0.25 * cov_s + 0.2 * contrast + 0.15 * center


_clip = None


def _feat(x):
    """transformers 4 devuelve un tensor; transformers 5, un objeto con pooler_output (ya proyectado)."""
    return x if hasattr(x, "norm") else x.pooler_output


def _load_clip():
    global _clip
    if _clip is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        _clip = (CLIPModel.from_pretrained(name, torch_dtype=torch.float32).eval(), CLIPProcessor.from_pretrained(name))
    return _clip


def relevance(images: list[Image.Image], text: str) -> list[float] | None:
    try:
        import torch
        model, proc = _load_clip()
        scores = []
        with torch.no_grad():
            t = proc(text=[f"a picture of {text}", "text, a logo, a screenshot, a collage"], return_tensors="pt",
                     padding=True)
            te = _feat(model.get_text_features(**t))
            te = te / te.norm(dim=-1, keepdim=True)
            for i in range(0, len(images), 16):
                v = proc(images=images[i:i + 16], return_tensors="pt")
                ie = _feat(model.get_image_features(**v))
                ie = ie / ie.norm(dim=-1, keepdim=True)
                sim = (ie @ te.T) * 100
                p = sim.softmax(-1)[:, 0]                   # prob. de ser el sujeto y no texto/logo
                raw = (sim[:, 0] - 18) / 14                 # coseno*100 ~18 (nada) .. ~32 (muy relevante)
                scores += (0.5 * p + 0.5 * raw.clamp(0, 1)).tolist()
        return scores
    except Exception:
        return None


def rank(pairs, text: str, log=None):
    if not pairs:
        return []
    ims = [p[1] for p in pairs]
    rel = relevance(ims, text)
    if rel is None:
        _log(log, "  · CLIP no disponible: se ordena solo por facilidad de pixelado")
        rel = [0.5] * len(ims)
    ranked = []
    for (it, im), r in zip(pairs, rel):
        px = pixelability(im)
        it = dict(it, relevancia=round(r, 3), pixelable=round(px, 3), score=round(0.65 * r + 0.35 * px, 3))
        ranked.append((it, im))
    ranked.sort(key=lambda x: -x[0]["score"])
    return ranked


# ------------------------------------------------------------------ recorte del sujeto

_rembg = None


def cutout(im: Image.Image) -> Image.Image:
    """Recorta el sujeto sobre fondo blanco con rembg (si está instalado). Si falla, devuelve la imagen tal cual."""
    global _rembg
    try:
        from rembg import new_session, remove
        if _rembg is None:
            _rembg = new_session("u2net")
        rgba = remove(im, session=_rembg)
        if np.asarray(rgba)[..., 3].mean() < 255 * 0.02:   # no encontró sujeto
            return im
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    except Exception:
        return im


def unload():
    """Libera CLIP y rembg (RAM/VRAM) antes de lanzar la IA de vídeo."""
    global _clip, _rembg
    _clip = None
    _rembg = None
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------------------ agente completo

def find_references(text: str, k: int = 4, outdir: str | None = None, use_llm: bool = True, social: bool = True,
                    cut: bool = True, log=None):
    """-> (lista de (meta, imagen_preparada), sujeto_en_inglés)."""
    t0 = time.time()
    qs = make_queries(text, 4, use_llm)
    subject = qs[0]
    _log(log, f"Consultas: {', '.join(qs[1:] or qs)}")
    items = search(qs[1:] or qs, per_query=12, social=social, log=log)
    _log(log, f"{len(items)} candidatas encontradas")
    if not items:
        return [], subject
    pairs = download(items, 60, log)
    ranked = rank(pairs, subject, log)[:k]
    res = []
    for i, (meta, im) in enumerate(ranked):
        prepared = cutout(im) if cut else im
        res.append((meta, prepared))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            im.save(os.path.join(outdir, f"ref{i + 1}_original.jpg"), quality=92)
            prepared.save(os.path.join(outdir, f"ref{i + 1}.png"))
    if outdir:
        with open(os.path.join(outdir, "referencias.json"), "w", encoding="utf-8") as fh:
            json.dump([m for m, _ in res], fh, ensure_ascii=False, indent=1)
    unload()
    _log(log, f"Mejores {len(res)} referencias en {time.time() - t0:.0f}s")
    return res, subject


def guess_motion(text: str) -> str:
    t = text.lower()
    rules = [(("bat", "murci", "bird", "pájaro", "pajaro", "raven", "cuervo", "crow", "eagle", "águila", "butterfly",
               "mariposa", "dragon", "dragón", "owl", "búho", "fly", "vol", "wing", "ala"), "aleteo"),
             (("ghost", "fantasma", "jelly", "medusa", "spirit", "espíritu", "float", "flot", "fish", "pez"), "flotar"),
             (("fire", "fuego", "flame", "llama", "smoke", "humo", "hair", "pelo", "flag", "bandera", "tree", "árbol"),
              "viento"),
             (("heart", "corazón", "corazon", "breath", "respir", "skull", "calavera", "sleep", "dorm"), "respirar"),
             (("dance", "baila", "walk", "camina", "run", "corr", "cat", "gato", "dog", "perro", "fox", "zorro", "horse", "caballo", "swing", "balance"), "balanceo")]
    for keys, m in rules:
        if any(k in t for k in keys):
            return m
    return "flotar"


# ------------------------------------------------------------------ ejecución aislada

def find_references_isolated(text: str, k: int = 4, outdir: str | None = None, use_llm: bool = True,
                             social: bool = True, cut: bool = True, log=None, timeout: float = 600):
    """Igual que find_references pero en un proceso Python aparte (subprocess): al terminar se libera toda la
    memoria (CLIP, rembg, descargas) antes de cargar la IA de vídeo. Devuelve (lista de (meta, imagen), sujeto)."""
    import subprocess
    import sys
    if outdir is None:
        import tempfile
        outdir = tempfile.mkdtemp(prefix="pixelanim-refs-")
    os.makedirs(outdir, exist_ok=True)
    args = [sys.executable, "-u", os.path.abspath(__file__), "--texto", text, "-k", str(k), "--out", outdir]
    if not use_llm:
        args.append("--sin-llm")
    if not social:
        args.append("--sin-redes")
    if not cut:
        args.append("--sin-recorte")
    _log(log, "Agente buscador: buscando y puntuando imágenes en internet…")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", cwd=os.path.dirname(os.path.abspath(__file__)), creationflags=flags)
    t0 = time.time()
    subject = text
    try:
        for line in proc.stdout:                       # progreso en directo
            line = line.rstrip()
            if line.startswith("@@SUBJECT "):
                subject = line[10:]
            elif line.startswith("  ·") or line.startswith(("Consultas", "Descargadas", "Mejores", "  ")) or "candidatas" in line:
                _log(log, line)
            if time.time() - t0 > timeout:
                proc.kill()
                raise TimeoutError("la búsqueda tardó demasiado")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    meta_path = os.path.join(outdir, "referencias.json")
    if proc.returncode != 0 and not os.path.exists(meta_path):
        raise RuntimeError("el agente buscador falló (mira la consola)")
    metas = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else []
    out = []
    for i, meta in enumerate(metas):
        p = os.path.join(outdir, f"ref{i + 1}.png")
        if os.path.exists(p):
            im = Image.open(p)
            im.load()
            out.append((meta, im.convert("RGB")))
    return out, subject


if __name__ == "__main__":
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto", required=True)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sin-llm", action="store_true")
    ap.add_argument("--sin-redes", action="store_true")
    ap.add_argument("--sin-recorte", action="store_true")
    a = ap.parse_args()
    res, subj = find_references(a.texto, a.k, a.out, not a.sin_llm, not a.sin_redes, not a.sin_recorte,
                                log=lambda m: print(m, flush=True))
    print("@@SUBJECT " + subj, flush=True)
    if not res:
        with open(os.path.join(a.out, "referencias.json"), "w", encoding="utf-8") as fh:
            json.dump([], fh)
