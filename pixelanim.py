"""PixelAnim — generador local de sprites animados tramados (box-shadow CSS, GIF, sprite sheet, fondo de pantalla).

Uso:
  python pixelanim.py web                                   (interfaz en el navegador)
  python pixelanim.py gen "murciélago volando" --res 96 --paleta gris
  python pixelanim.py gen "búho volando" --web --variantes 3    (busca fotos en internet y las anima)
  python pixelanim.py gen "ghost floating" --estilo sombra --semillas 1 2 3 --sin-llm
  python pixelanim.py imagen foto.png --movimiento aleteo --fuerza 0.6
  python pixelanim.py refs "búho volando"                   (solo buscar y guardar referencias)
  python pixelanim.py video clip.mp4 --res 128 --paleta fosforo
  python pixelanim.py probar [--ia]                         (comprobación de la instalación)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from PIL import Image, ImageSequence

import motor
from motor import OUT, slug  # noqa: F401  (reexportados para app.py)


def read_media(path: str, max_frames: int = 24, seconds: float = 3.0) -> list[Image.Image]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gif", ".webp", ".png", ".jpg", ".jpeg", ".bmp"):
        im = Image.open(path)
        fr = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    else:
        import imageio.v2 as iio  # requiere imageio-ffmpeg
        try:
            rd = iio.get_reader(path, "ffmpeg")
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir el vídeo ({e}). Prueba con MP4/WebM o conviértelo a GIF.") from e
        fps = rd.get_meta_data().get("fps", 30) or 30
        fr = []
        try:
            for i, a in enumerate(rd):
                if i >= seconds * fps:
                    break
                fr.append(Image.fromarray(a))
        finally:
            rd.close()
    if len(fr) > max_frames:
        step = len(fr) / max_frames
        fr = [fr[int(i * step)] for i in range(max_frames)]
    return fr


def pix_kwargs(a):
    return dict(res=a.res, paleta=a.paleta, tonos=a.tonos, grano=a.grano, motas=a.motas, contraste=a.contraste,
                umbral=a.umbral, zoom=a.zoom, ms=a.ms)


def show(name, info):
    print("  ->", info)


def add_pix_args(p):
    import pixelize as px
    p.add_argument("--res", type=int, default=96, help="resolución del sprite (64, 96, 128, 160)")
    p.add_argument("--paleta", default="gris", choices=list(px.PALETTES))
    p.add_argument("--tonos", type=int, default=3, help="tonos de tinta (1 = 1-bit puro)")
    p.add_argument("--grano", type=float, default=0.35, help="ruido que cambia en cada fotograma (0-1)")
    p.add_argument("--motas", type=float, default=0.6, help="píxeles sueltos en el borde (0-2)")
    p.add_argument("--contraste", type=float, default=1.3)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--zoom", type=float, default=1.0, help=">1 acerca el sujeto")
    p.add_argument("--ms", type=int, default=80, help="ms por fotograma")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generar con IA local a partir de texto")
    g.add_argument("texto")
    g.add_argument("--estilo", default="silueta")
    g.add_argument("--semillas", type=int, nargs="+", default=[0])
    g.add_argument("--web", action="store_true", help="nutrirse de imágenes de internet (agente buscador)")
    g.add_argument("--variantes", type=int, default=3, help="con --web: cuántas referencias animar")
    g.add_argument("--sin-llm", dest="llm", action="store_false", help="no usar Ollama")
    add_pix_args(g)

    im = sub.add_parser("imagen", help="animar una imagen fija")
    im.add_argument("archivo")
    im.add_argument("--texto", default="", help="qué es / cómo se mueve (vacío = autodescripción con Ollama)")
    im.add_argument("--estilo", default="silueta")
    im.add_argument("--movimiento", default="flotar", help="flotar, respirar, balanceo, viento, aleteo, ninguno")
    im.add_argument("--fuerza", type=float, default=0.6, help="0 = sin IA · 0.4 fiel · 0.8 reinterpreta")
    im.add_argument("--semillas", type=int, nargs="+", default=[0])
    im.add_argument("--sin-llm", dest="llm", action="store_false")
    add_pix_args(im)

    r = sub.add_parser("refs", help="solo buscar y guardar imágenes de referencia")
    r.add_argument("texto")
    r.add_argument("-n", type=int, default=6)
    r.add_argument("--sin-llm", dest="llm", action="store_false")

    v = sub.add_parser("video", help="pixelizar un vídeo/GIF/imagen existente")
    v.add_argument("archivo")
    v.add_argument("--max", type=int, default=24)
    add_pix_args(v)

    t = sub.add_parser("probar", help="comprobar que todo está bien instalado")
    t.add_argument("--ia", action="store_true", help="incluir una generación real (descarga modelos la 1ª vez)")

    sub.add_parser("web", help="interfaz web local")
    a = ap.parse_args(argv)

    if a.cmd == "web":
        import app
        app.launch()
        return 0
    if a.cmd == "probar":
        import probar
        return probar.main(a.ia)
    if a.cmd == "refs":
        import buscador
        d = os.path.join(OUT, motor.unique_name(f"refs-{slug(a.texto)}"))
        res, subj = buscador.find_references(a.texto, a.n, d, a.llm, log=print)
        for i, (m, _) in enumerate(res, 1):
            print(f"  ref{i}: {m['score']:.2f}  {m['source']:<10} {m['url'][:90]}")
        print(f"  -> {d}")
        return 0
    if a.cmd == "video":
        fr = read_media(a.archivo, a.max)
        name = motor.unique_name(slug(os.path.splitext(os.path.basename(a.archivo))[0]))
        _, _, info = motor.render(fr, name, **pix_kwargs(a))
        show(name, info)
        return 0

    import generate as gen
    print("Dispositivo:", gen.device_info())
    if a.cmd == "imagen":
        src = Image.open(a.archivo)
        src.load()
        for s in a.semillas:
            t0 = time.time()
            frames, prompt, name = motor.run_image(src, a.texto, a.estilo, a.movimiento, a.fuerza, s, a.llm,
                                                   label=os.path.splitext(os.path.basename(a.archivo))[0])
            print(f"[{s}] {time.time() - t0:.0f}s  prompt: {prompt}")
            show(name, motor.render(frames, name, **pix_kwargs(a))[2])
        return 0

    for s in a.semillas:
        t0 = time.time()
        if a.web:
            for frames, prompt, name, meta in motor.run_web(a.texto, a.estilo, s, a.llm, a.variantes, log=print):
                show(name, motor.render(frames, name, **pix_kwargs(a))[2])
            print(f"[{s}] {time.time() - t0:.0f}s")
        else:
            frames, prompt, name = motor.run_text(a.texto, a.estilo, s, a.llm)
            print(f"[{s}] {time.time() - t0:.0f}s  prompt: {prompt}")
            show(name, motor.render(frames, name, **pix_kwargs(a))[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
