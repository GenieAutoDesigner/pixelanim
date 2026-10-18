"""Comprobación de la instalación: python pixelanim.py probar [--ia]"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback

OK, BAD, WARN = "OK   ", "FALLO", "AVISO"
results = []


def check(name, fn, optional=False):
    t = time.time()
    try:
        msg = fn() or ""
        results.append((OK, name, msg))
    except Exception as e:  # noqa: BLE001
        results.append((WARN if optional else BAD, name, f"{type(e).__name__}: {e}"))
        if not optional:
            traceback.print_exc()
    print(f"[{results[-1][0]}] {name} ({time.time() - t:.1f}s) {results[-1][2]}", flush=True)


def t_imports():
    import diffusers, gradio, numpy, PIL, torch, transformers  # noqa: F401,E401
    return f"torch {torch.__version__} · diffusers {diffusers.__version__} · transformers {transformers.__version__} · gradio {gradio.__version__}"


def t_gpu():
    import generate
    return generate.device_info()


def t_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _frames(n=12):
    from PIL import Image, ImageDraw
    out = []
    for i in range(n):
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        r = 50 + (i % 6) * 4
        d.ellipse((128 - r, 100 - r + i * 3, 128 + r, 100 + r + i * 3), fill="black")
        out.append(im)
    return out


def t_pixel():
    import motor
    tmp = tempfile.mkdtemp()
    old = motor.OUT
    motor.OUT = tmp
    try:
        gif, files, info = motor.render(_frames(), "prueba", res=64, paleta="fosforo", tonos=2)
        assert all(os.path.getsize(f) > 0 for f in files), "algún archivo vacío"
        motor.render([_frames(1)[0]], "una-imagen", res=48)          # 1 solo fotograma
        from PIL import Image
        motor.render([Image.new("RGB", (64, 64), "white")] * 4, "vacio", res=48)   # fotogramas vacíos
        return info.split(" · " + tmp)[0]
    finally:
        motor.OUT = old
        shutil.rmtree(tmp, ignore_errors=True)


def t_chat():
    import chat
    a = chat.Agent()
    seq = [("un cuervo volando", "generar"), ("en verde y más grande", "reajustar"), ("otra versión", "generar"),
           ("hola", "nada")]
    for msg, want in seq:
        _, act, p, via = a.step(msg)
        a.has_result = True
        if via == "reglas" and act != want:
            raise AssertionError(f"«{msg}» -> {act}, esperaba {want}")
    return f"vía {via}"


def t_ollama():
    import llm
    m = llm.text_model()
    if not m:
        raise RuntimeError("Ollama no está abierto (opcional: el chat usará reglas simples)")
    out = llm.chat_json([{"role": "user", "content": 'Answer only JSON {"ok": true}'}])
    if not isinstance(out, dict):
        raise RuntimeError(f"{m} no devolvió JSON")
    vm = llm.vision_model()
    return f"texto: {m} · visión: {vm or 'ninguno (opcional)'}"


def t_web():
    import buscador
    items = buscador.search(["raven silhouette"], per_query=6, social=False)
    if not items:
        raise RuntimeError("sin resultados (¿sin internet o bloqueado?)")
    pairs = buscador.download(items, 8)
    if not pairs:
        raise RuntimeError("no se pudo descargar ninguna imagen")
    srcs = sorted({i["source"] for i in items})
    return f"{len(items)} resultados de {', '.join(srcs)} · {len(pairs)} descargadas"


def t_clip():
    import buscador
    from PIL import Image
    s = buscador.relevance([Image.new("RGB", (224, 224), "white")], "a raven")
    if s is None:
        raise RuntimeError("CLIP no cargó (se ordenará solo por pixelabilidad)")
    return "relevancia CLIP disponible"


def t_rembg():
    import buscador
    from PIL import Image
    buscador.cutout(_frames(1)[0])
    import rembg  # noqa: F401
    return "recorte de sujeto disponible"


def t_ia_text():
    import generate
    fr, _ = generate.generate("a bat flapping its wings", "silueta", 8, 1, 256, 4, False)
    assert len(fr) == 8
    return "8 fotogramas a 256 px"


def t_ia_image():
    import generate
    fr, _ = generate.animate_image(_frames(1)[0], "a black ball", "silueta", "flotar", 0.5, 8, 1, 256, False)
    assert len(fr) == 8
    return "8 fotogramas a 256 px"


def main(ia=False):
    print("PixelAnim · comprobación\n")
    check("Librerías", t_imports)
    check("Dispositivo", t_gpu)
    check("ffmpeg (vídeos)", t_ffmpeg, optional=True)
    check("Pixelado y exportación", t_pixel)
    check("Chat (reglas)", t_chat)
    check("Ollama", t_ollama, optional=True)
    check("Búsqueda en internet", t_web, optional=True)
    check("CLIP (relevancia de referencias)", t_clip, optional=True)
    check("rembg (recorte del sujeto)", t_rembg, optional=True)
    try:
        import buscador
        buscador.unload()
    except Exception:
        pass
    if ia:
        check("IA: texto → animación", t_ia_text)
        check("IA: imagen → animación", t_ia_image)
    bad = [r for r in results if r[0] == BAD]
    warn = [r for r in results if r[0] == WARN]
    print(f"\n{len(results) - len(bad) - len(warn)} OK · {len(warn)} avisos · {len(bad)} fallos")
    if not ia and not bad:
        print("Para probar también la IA (descarga ~2,5 GB la 1ª vez): python pixelanim.py probar --ia")
    return 1 if bad else 0
