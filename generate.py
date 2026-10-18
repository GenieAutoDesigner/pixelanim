"""Generación local de animaciones con AnimateDiff-Lightning (SD 1.5 + módulo de movimiento, 4 pasos).

Primera ejecución: descarga ~2,9 GB de pesos a la caché de Hugging Face (una sola vez).
GPU NVIDIA: 6 GB VRAM mínimo recomendado. Sin GPU funciona en CPU, pero tarda minutos por clip.
"""
from __future__ import annotations

import math
import os

import torch
from diffusers import AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline, EulerDiscreteScheduler, MotionAdapter
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_MODEL = os.environ.get("PIXELANIM_BASE", "Lykon/dreamshaper-8")
LIGHTNING_REPO = "ByteDance/AnimateDiff-Lightning"

STYLES = {
    "silueta": ("{s}, solid pitch black silhouette, isolated on plain white background, high contrast, "
                "minimalist, centered, full body, flat black shape, no details"),
    "tinta":   ("{s}, black ink illustration, bold brush strokes, isolated on plain white paper, "
                "high contrast, centered, monochrome"),
    "sombra":  ("{s}, dark shadowy figure, black smoke and soft edges, plain white background, "
                "monochrome, centered, eerie"),
    "detalle": ("{s}, detailed black and white illustration, strong lighting, plain white background, "
                "centered, monochrome, sharp"),
}
NEGATIVE = ("text, watermark, signature, frame, border, gradient background, scenery, landscape, "
            "multiple subjects, cropped, blurry, color, grey background")

_pipe = None
_v2v = None


def device_info() -> str:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"GPU {p.name} ({p.total_memory / 2**30:.1f} GB)"
    return "CPU (sin GPU CUDA: será lento)"


def _trim():
    """Devuelve al sistema la RAM liberada tras la carga (importante con 8-16 GB de RAM)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def load_pipeline(steps: int = 4):
    global _pipe
    if _pipe is not None:
        return _pipe
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.bfloat16  # bf16 en CPU: mitad de RAM
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # evita construir el UNet de movimiento en fp32 (pico de ~5 GB de RAM)
    try:
        adapter = MotionAdapter()
        ckpt = hf_hub_download(LIGHTNING_REPO, f"animatediff_lightning_{steps}step_diffusers.safetensors")
        sd = load_file(ckpt, device="cpu")
        adapter.load_state_dict(sd)
        del sd
        pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL, motion_adapter=adapter, torch_dtype=dtype,
                                                   variant="fp16")
    finally:
        torch.set_default_dtype(prev)
    _trim()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing",
                                                        beta_schedule="linear")
    pipe.vae.enable_slicing()
    global _offload
    if cuda:
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        _offload = vram < 7.5
        if not _offload:
            pipe.to("cuda")
    _pipe = pipe
    return pipe


_offload = False
_active = None
_lock = __import__("threading").Lock()   # una sola generación a la vez (la GPU no se comparte)


def _guarded(fn):
    """Serializa las generaciones y, si la GPU se queda sin memoria, activa el offload y reintenta una vez."""
    import functools

    @functools.wraps(fn)
    def wrap(*a, **k):
        global _offload, _active
        with _lock:
            try:
                return fn(*a, **k)
            except torch.cuda.OutOfMemoryError:
                if not torch.cuda.is_available() or _offload:
                    raise
                torch.cuda.empty_cache()
                for p in (_pipe, _v2v):
                    if p is not None:
                        p.to("cpu")
                torch.cuda.empty_cache()
                _offload, _active = True, None
                return fn(*a, **k)
    return wrap


def _activate(pipe):
    """Con poca VRAM, el offload CPU<->GPU se registra en la canalización que se va a usar (comparten módulos)."""
    global _active
    if _offload and _active is not pipe:
        if _active is not None:
            _active.remove_all_hooks()
        pipe.enable_model_cpu_offload()
        _active = pipe
    return pipe


from llm import enhance_prompt  # noqa: E402


@_guarded
def generate(subject: str, style: str = "silueta", frames: int = 16, seed: int = 0, size: int = 512,
             steps: int = 4, use_llm: bool = True, progress=None):
    """Devuelve (lista de PIL.Image, prompt usado)."""
    pipe = _activate(load_pipeline(steps))
    subj = enhance_prompt(subject) if use_llm else subject
    prompt = STYLES.get(style, STYLES["silueta"]).format(s=subj)
    g = torch.Generator("cpu").manual_seed(int(seed))
    kw = {}
    if progress:
        kw["callback_on_step_end"] = lambda p, i, t, d: (progress((i + 1) / steps), d)[1]
    out = pipe(prompt=prompt, negative_prompt=NEGATIVE, num_frames=frames, guidance_scale=1.0,
               num_inference_steps=steps, width=size, height=size, generator=g, decode_chunk_size=4, **kw)
    return out.frames[0], prompt


# ------------------------------------------------------------------ imagen -> animación

MOTIONS = ("flotar", "respirar", "balanceo", "viento", "aleteo", "ninguno")


def _prep(img: Image.Image, size: int) -> Image.Image:
    """Encaja la imagen en un cuadrado con margen, rellenando con el color de sus bordes."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    px = img.load()
    edge = [px[x, y] for x in range(0, w, max(1, w // 20)) for y in (0, h - 1)] + \
           [px[x, y] for y in range(0, h, max(1, h // 20)) for x in (0, w - 1)]
    bg = tuple(sorted(c[i] for c in edge)[len(edge) // 2] for i in range(3))
    # recortar al sujeto (si el fondo es liso), para que ocupe el encuadre
    try:
        import numpy as np
        from pixelize import clean_mask
        small = img.resize((128, max(1, int(128 * h / w))), Image.Resampling.BOX)
        d = np.abs(np.asarray(small, np.float32) - np.array(bg, np.float32)).max(-1)
        m = clean_mask(d > 40)
        if m.sum() > 30:
            ys, xs = np.nonzero(m)
            sx, sy = w / small.width, h / small.height
            x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
            y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
            if (x1 - x0) * (y1 - y0) < 0.8 * w * h:
                mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
                img = img.crop((int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))))
                w, h = img.size
    except Exception:
        pass
    side = int(max(w, h) * 1.15)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS), bg


def motion_frames(img: Image.Image, n: int = 16, motion: str = "flotar", amount: float = 1.0, size: int = 512):
    """Vídeo base periódico a partir de una imagen fija (bucle perfecto: t = i/n)."""
    base, bg = _prep(img, size)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        a = amount
        if motion == "flotar":
            dx, dy, rot, sc = 0, math.sin(t) * 0.035 * size * a, math.sin(t + 1) * 2 * a, 1
        elif motion == "respirar":
            dx, dy, rot, sc = 0, 0, 0, 1 + math.sin(t) * 0.045 * a
        elif motion == "balanceo":
            dx, dy, rot, sc = 0, 0, math.sin(t) * 7 * a, 1
        elif motion == "aleteo":
            dx, dy, rot, sc = 0, math.sin(t) * 0.02 * size * a, 0, 1 + abs(math.sin(t)) * 0.06 * a
        else:
            dx = dy = rot = 0
            sc = 1
        f = base
        if sc != 1:
            s2 = int(size * sc)
            f = f.resize((s2, s2), Image.Resampling.BICUBIC)
            c = Image.new("RGB", (size, size), bg)
            c.paste(f, ((size - s2) // 2, (size - s2) // 2))
            f = c
        if rot:
            f = f.rotate(rot, Image.Resampling.BICUBIC, fillcolor=bg)
        if dx or dy:
            f = f.transform(f.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy), Image.Resampling.BICUBIC, fillcolor=bg)
        if motion == "viento":  # ondulación horizontal por filas
            import numpy as np
            arr = np.asarray(f).copy()
            for y in range(size):
                arr[y] = np.roll(arr[y], int(math.sin(t + y / size * 6) * 0.02 * size * a * (y / size)), axis=0)
            f = Image.fromarray(arr)
        out.append(f)
    return out


def load_v2v():
    global _v2v
    if _v2v is None:
        p = load_pipeline()
        # mismos módulos en memoria (from_pipe duplicaba el UNet y se quedaba sin RAM)
        _v2v = AnimateDiffVideoToVideoPipeline(vae=p.vae, text_encoder=p.text_encoder, tokenizer=p.tokenizer,
                                               unet=p.unet, motion_adapter=p.motion_adapter, scheduler=p.scheduler,
                                               feature_extractor=p.feature_extractor, image_encoder=p.image_encoder)
    return _v2v


@_guarded
def animate_image(img: Image.Image, subject: str = "", style: str = "silueta", motion: str = "flotar",
                  strength: float = 0.6, frames: int = 16, seed: int = 0, size: int = 512, use_llm: bool = True,
                  amount: float = 1.0, progress=None):
    """Imagen fija -> animación. strength 0.4 = fiel a la imagen · 0.8 = la IA reinterpreta más.
    Devuelve (fotogramas, prompt). Con strength 0 no usa IA (solo movimiento procedural)."""
    base = motion_frames(img, frames, motion, amount, size)
    if strength <= 0:
        return base, "(procedural, sin IA)"
    if not subject.strip():
        import llm
        subject = (llm.describe_image(img) if use_llm else None) or "the subject"
    elif use_llm:
        subject = enhance_prompt(subject)
    prompt = STYLES.get(style, STYLES["silueta"]).format(s=subject)
    pipe = _activate(load_v2v())
    steps = max(4, round(4 / max(strength, 0.2)))   # ~4 pasos efectivos (modelo Lightning de 4 pasos)
    g = torch.Generator("cpu").manual_seed(int(seed))
    kw = {}
    if progress:
        kw["callback_on_step_end"] = lambda p, i, t, d: (progress(min(1.0, (i + 1) / max(1, steps * strength))), d)[1]
    out = pipe(video=base, prompt=prompt, negative_prompt=NEGATIVE, strength=strength, guidance_scale=1.0,
               height=size, width=size,
               num_inference_steps=steps, generator=g, decode_chunk_size=2, **kw)
    return out.frames[0], prompt
