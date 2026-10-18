"""Orquestación común para la interfaz web y la consola: generar, animar imagen, buscar referencias, pixelar y exportar."""
from __future__ import annotations

import os
import re
import unicodedata

from PIL import Image

import pixelize as px

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "salidas")
# Resolución de trabajo de la IA. 512 = calidad normal. Con poca VRAM/RAM: set PIXELANIM_SIZE=384
SIZE = int(os.environ.get("PIXELANIM_SIZE", "512"))
PIX_KEYS = ["res", "paleta", "tonos", "grano", "motas", "contraste", "umbral", "zoom", "ms"]


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-")
    return s if s and s[0].isalpha() else "s-" + (s or "sprite")


def unique_name(base: str) -> str:
    name, i = base, 2
    while os.path.exists(os.path.join(OUT, name)):
        name = f"{base}-{i}"
        i += 1
    return name


def save_raw(frames, name):
    d = os.path.join(OUT, name, "ia")
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(d, f"{i:02d}.png"))


def render(frames, name, res=96, paleta="gris", tonos=3, grano=0.35, motas=0.6, contraste=1.3, umbral=0.5,
           zoom=1.0, ms=80):
    """Pixela y exporta. -> (gif, [archivos], texto_info)"""
    if not frames:
        raise ValueError("No hay fotogramas que pixelar")
    idx = px.pixelize(frames, res=int(res), levels=int(tonos), grain=float(grano), specks=float(motas),
                      contrast=float(contraste), threshold=float(umbral), zoom=float(zoom))
    d = os.path.join(OUT, name)
    info = px.export_all(idx, d, name, paleta, int(tonos), int(ms))
    files = [os.path.join(d, f) for f in (f"{name}.gif", f"{name}.css", f"{name}_css.html",
                                          f"{name}_wallpaper.html", f"{name}_sheet.png")]
    return files[0], files, f"{info['frames']} fotogramas · {info['res']}×{info['res']} · CSS {info['css_kb']} KB · {d}"


def run_text(tema, estilo="silueta", semilla=0, use_llm=True, progress=None, size=None):
    import generate as gen
    frames, prompt = gen.generate(tema, estilo, 16, int(semilla), size or SIZE, 4, use_llm, progress=progress)
    name = unique_name(f"{slug(tema)}-{int(semilla)}")
    save_raw(frames, name)
    return frames, prompt, name


def run_image(img, tema="", estilo="silueta", movimiento="flotar", fuerza=0.6, semilla=0, use_llm=True,
              progress=None, size=None, label=None):
    import generate as gen
    frames, prompt = gen.animate_image(img, tema, estilo, movimiento, float(fuerza), 16, int(semilla), size or SIZE,
                                       use_llm, progress=progress)
    name = unique_name(f"img-{slug(label or tema or 'imagen')}-{int(semilla)}")
    save_raw(frames, name)
    return frames, prompt, name


def run_web(tema, estilo="silueta", semilla=0, use_llm=True, n=3, fuerza=0.7, movimiento=None, log=None,
            progress=None, size=None, social=True):
    """Texto -> busca referencias en internet -> anima las n mejores. -> lista de (frames, prompt, name, meta)."""
    import buscador
    base = unique_name(f"web-{slug(tema)}-{int(semilla)}")
    refdir = os.path.join(OUT, base, "referencias")
    refs, subject = buscador.find_references_isolated(tema, k=n, outdir=refdir, use_llm=use_llm, social=social,
                                                      log=log)
    if not refs:
        raise RuntimeError("No se encontraron imágenes de referencia (¿sin conexión?). Prueba sin «internet».")
    mov = movimiento or buscador.guess_motion(f"{tema} {subject}")
    out = []
    for i, (meta, im) in enumerate(refs):
        if log:
            log(f"Animando referencia {i + 1}/{len(refs)} ({meta.get('source')}, puntuación {meta.get('score')}) · {mov}")
        sub_prog = (lambda f, i=i: progress((i + f) / len(refs))) if progress else None
        frames, prompt, name = run_image(im, subject, estilo, mov, fuerza, semilla, False, sub_prog, size,
                                         label=f"{base}-r{i + 1}")
        out.append((frames, prompt, name, dict(meta, _base=base)))
    return out
