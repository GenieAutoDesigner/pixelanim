"""Interfaz web local (Gradio). Abre http://127.0.0.1:7860"""
from __future__ import annotations

import functools
import glob
import inspect
import os
import time
import traceback

import gradio as gr
from PIL import Image

import chat as chatmod
import llm
import motor
import pixelize as px
from motor import PIX_KEYS, render, slug
from pixelanim import read_media


class Session:
    def __init__(self):
        self.frames = None
        self.name = None
        self.image = None          # última imagen adjunta en el chat
        self.results = []          # [(frames, name, gif)] de la última tanda
        self.agent = chatmod.Agent()


S = Session()


def _friendly(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if "out of memory" in low or "outofmemory" in low:
        return "Sin memoria en la GPU/RAM. Cierra otros programas (o el modelo de Ollama) y vuelve a intentarlo."
    if any(k in low for k in ("connection", "urlopen", "httperror", "max retries", "name resolution")):
        return "Fallo de conexión al descargar modelos o imágenes. Comprueba internet y reintenta."
    return msg[:400]


# ------------------------------------------------------------------ cierre automático
# Al cerrar la pestaña del navegador se apaga el servidor (libera el puerto) y el Ollama que hayamos arrancado.
# Nunca se apaga en mitad de una generación.

import threading  # noqa: E402

IDLE_AFTER_CLOSE = 8      # s sin pestañas abiertas -> salir
IDLE_NO_BEAT = 180        # s sin latidos (navegador cerrado de golpe / pestaña congelada) -> salir
_life = {"sessions": set(), "seen": False, "beat": time.time(), "empty_since": None, "busy": 0}
_life_lock = threading.Lock()


class _Busy:
    def __enter__(self):
        with _life_lock:
            _life["busy"] += 1

    def __exit__(self, *exc):
        with _life_lock:
            _life["busy"] -= 1
            _life["beat"] = time.time()


def _hello(request: gr.Request = None):
    with _life_lock:
        _life["seen"] = True
        _life["beat"] = time.time()
        _life["empty_since"] = None
        sid = getattr(request, "session_hash", None)
        if sid:
            _life["sessions"].add(sid)


def _bye(request: gr.Request = None):
    with _life_lock:
        sid = getattr(request, "session_hash", None)
        _life["sessions"].discard(sid)
        if not _life["sessions"]:
            _life["empty_since"] = time.time()


def _shutdown(reason):
    print(f"\nPixelAnim se cierra ({reason}). Puerto liberado.", flush=True)
    try:
        llm.stop_started()
    except Exception:
        pass
    os._exit(0)


def _watchdog():
    while True:
        time.sleep(2)
        with _life_lock:
            if not _life["seen"] or _life["busy"]:
                continue
            now = time.time()
            if _life["empty_since"] and now - _life["empty_since"] > IDLE_AFTER_CLOSE:
                reason = "pestaña cerrada"
            elif now - _life["beat"] > IDLE_NO_BEAT:
                reason = "sin navegador"
            else:
                continue
        _shutdown(reason)


def safe(fn):
    """Cualquier excepción se muestra en la interfaz en vez de colgarla."""
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            with _Busy():
                return fn(*a, **k)
        except gr.Error:
            raise
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise gr.Error(_friendly(e)) from e
    return wrap


def _set_results(items):
    """items: [(frames, name, gif)] -> valor de la galería de variantes."""
    S.results = list(items)
    S.frames, S.name = items[0][0], items[0][1]
    return [(g, n) for _, n, g in S.results]


def _ref_gallery(base):
    d = os.path.join(motor.OUT, base, "referencias")
    return [(f, os.path.basename(f)) for f in sorted(glob.glob(os.path.join(d, "ref*.png")))]


def _render_all(items, pix):
    out = []
    for fr, nm, _ in items:
        g, _, _ = render(fr, nm, *pix)
        out.append((fr, nm, g))
    return out


def _keep_choice(cur):
    for fr, nm, _ in S.results:
        if nm == cur:
            S.frames, S.name = fr, nm


# ------------------------------------------------------------------ pestañas

@safe
def do_text(texto, estilo, semilla, use_llm, web, nrefs, *pix, progress=gr.Progress()):
    if not (texto or "").strip():
        raise gr.Error("Escribe qué quieres animar")
    t = time.time()
    if web:
        progress(0, desc="Buscando referencias en internet")
        res = motor.run_web(texto, estilo, semilla, use_llm, int(nrefs), log=lambda m: progress(None, desc=m[:90]),
                            progress=lambda f: progress(f, desc="Animando referencias"))
        gal = _set_results(_render_all([(fr, nm, None) for fr, _, nm, _ in res], pix))
        gif, files, info = render(S.frames, S.name, *pix)
        return (gif, files, f"{time.time() - t:.0f}s · {len(res)} variantes desde internet · {res[0][1]}\n{info}",
                gal, _ref_gallery(res[0][3]["_base"]))
    progress(0, desc="Cargando modelo (la 1ª vez descarga ~2,5 GB)")
    frames, prompt, name = motor.run_text(texto, estilo, semilla, use_llm, lambda f: progress(f, desc="Generando"))
    gal = _set_results(_render_all([(frames, name, None)], pix))
    gif, files, info = render(frames, name, *pix)
    return gif, files, f"{time.time() - t:.0f}s · {prompt}\n{info}", gal, []


@safe
def do_image(img, texto, estilo, movimiento, fuerza, semilla, use_llm, *pix, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Sube una imagen")
    t = time.time()
    progress(0, desc="Preparando")
    src = Image.open(img)
    src.load()
    frames, prompt, name = motor.run_image(src, texto, estilo, movimiento, fuerza, semilla, use_llm,
                                           lambda f: progress(f, desc="Animando"),
                                           label=os.path.splitext(os.path.basename(img))[0])
    gal = _set_results(_render_all([(frames, name, None)], pix))
    gif, files, info = render(frames, name, *pix)
    return gif, files, f"{time.time() - t:.0f}s · {prompt}\n{info}", gal


@safe
def do_video(archivo, *pix):
    if not archivo:
        raise gr.Error("Sube un archivo")
    frames = read_media(archivo, 24)
    if not frames:
        raise gr.Error("No se pudo leer ningún fotograma del archivo")
    name = motor.unique_name(slug(os.path.splitext(os.path.basename(archivo))[0]))
    gal = _set_results(_render_all([(frames, name, None)], pix))
    gif, files, info = render(frames, name, *pix)
    return gif, files, info, gal


@safe
def do_retune(*pix):
    if S.frames is None:
        raise gr.Error("Genera o sube algo primero")
    S.agent.params.update(dict(zip(PIX_KEYS, pix)))
    cur = S.name
    gal = _set_results(_render_all(S.results or [(S.frames, S.name, None)], pix))
    _keep_choice(cur)
    gif, files, info = render(S.frames, S.name, *pix)
    return gif, files, info, gal


@safe
def do_pick(res, paleta, tonos, grano, motas, contraste, umbral, zoom, ms, evt: gr.SelectData):
    pix = [res, paleta, tonos, grano, motas, contraste, umbral, zoom, ms]
    if not S.results or evt.index is None:
        raise gr.Error("Nada que elegir")
    i = evt.index if isinstance(evt.index, int) else evt.index[0]
    if i >= len(S.results):
        raise gr.Error("Nada que elegir")
    frames, name, _ = S.results[i]
    S.frames, S.name = frames, name
    return render(frames, name, *pix)


# ------------------------------------------------------------------ chat

def _msg_parts(msg):
    if isinstance(msg, dict):
        files = []
        for f in msg.get("files", []) or []:
            files.append(f if isinstance(f, str) else (f.get("path") if isinstance(f, dict) else getattr(f, "path", None)))
        return msg.get("text", "") or "", [f for f in files if f]
    return str(msg or ""), []


def do_chat(msg, history, progress=gr.Progress()):
    history = list(history or [])
    try:
        with _Busy():
            return _do_chat(msg, history, progress)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        history.append({"role": "assistant", "content": "⚠ " + _friendly(e)})
        no = gr.update()
        return (gr.update(value=None), history, no, no, no, no, no, *[no] * len(PIX_KEYS))


def _do_chat(msg, history, progress):
    text, files = _msg_parts(msg)
    imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
    for f in imgs:
        history.append({"role": "user", "content": {"path": f}})
    if text:
        history.append({"role": "user", "content": text})
    if imgs:
        S.image = Image.open(imgs[0])
        S.image.load()
    prev_tema = S.agent.params["tema"]
    reply, action, p, via = S.agent.step(text, bool(imgs), S.image is not None)
    new_subject = (not imgs) and bool(p["tema"]) and p["tema"] != prev_tema
    use_image = S.image is not None and (bool(imgs) or ((S.name or "").startswith("img-") and not new_subject
                                                         and not (S.name or "").startswith("img-web-")))
    if p.get("web") and not imgs and p["tema"]:
        use_image = False
    if not use_image and not p["tema"].strip() and S.image is not None:
        use_image = True          # sin tema: solo puede ser la imagen
    no = gr.update()
    gif = files_out = gal = refs = info = no
    if action in ("generar", "reajustar"):
        t = time.time()
        pix = [p[k] for k in PIX_KEYS]
        use_llm = True   # el tema se pasa siempre por enhance_prompt (traduce al inglés si hace falta)
        prog = lambda f: progress(f, desc="Generando")  # noqa: E731
        cur = S.name
        if action == "generar":
            if use_image:
                frames, prompt, name = motor.run_image(S.image, p["tema"], p["estilo"], p["movimiento"], p["fuerza"],
                                                       p["semilla"], use_llm, prog)
                items = [(frames, name, None)]
            elif p.get("web"):
                res = motor.run_web(p["tema"], p["estilo"], p["semilla"], use_llm, 3,
                                    log=lambda m: progress(None, desc=m[:90]), progress=prog)
                items = [(fr, nm, None) for fr, _, nm, _ in res]
                prompt = res[0][1]
                refs = _ref_gallery(res[0][3]["_base"])
                reply = f"{reply} {len(res)} variantes a partir de fotos de internet; elige una en «Variantes»."
            else:
                frames, prompt, name = motor.run_text(p["tema"], p["estilo"], p["semilla"], use_llm, prog)
                items = [(frames, name, None)]
            gal = _set_results(_render_all(items, pix))
        else:
            prompt = "(mismo clip, nuevo pixelado)"
            gal = _set_results(_render_all(S.results or [(S.frames, S.name, None)], pix))
            _keep_choice(cur)
        gif, files_out, inf = render(S.frames, S.name, *pix)
        S.agent.has_result = True
        info = f"{time.time() - t:.0f}s · {prompt}\n{inf}"
        history.append({"role": "assistant", "content": reply})
        history.append({"role": "assistant", "content": {"path": gif}})
    else:
        history.append({"role": "assistant", "content": reply})
    sliders = [gr.update(value=p[k]) for k in PIX_KEYS]
    return (gr.update(value=None), history, gif, files_out, info, gal, refs, *sliders)


def do_chat_reset():
    S.agent = chatmod.Agent()
    S.image = None
    return []


# ------------------------------------------------------------------ UI

def _status():
    import generate as gen
    tm, vm = llm.text_model(), llm.vision_model()
    ol = (f"Ollama: chat con **{tm}**" + (f", visión con **{vm}**" if vm else ", sin modelo de visión")) if tm \
        else "Ollama no detectado: el chat entiende órdenes simples y conviene escribir el tema en inglés"
    return f"Dispositivo: **{gen.device_info()}** · {ol}"


def build():
    import generate as gen
    blocks_kw = {"title": "PixelAnim"}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw["theme"] = gr.themes.Monochrome()

    with gr.Blocks(**blocks_kw) as demo:
        gr.Markdown("# PixelAnim\nIA local → sprite tramado animado.  \n" + _status())
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(height=430, label="Asistente")
                    box = gr.MultimodalTextbox(file_types=["image"], file_count="single", show_label=False,
                                               placeholder="«un cuervo volando en rojo» · «busca en internet un búho» · adjunta una imagen · «más grande», «otra versión»…")
                    breset = gr.Button("Nueva conversación", size="sm")
                with gr.Tab("Texto"):
                    texto = gr.Textbox(label="Qué quieres", value="murciélago volando de frente")
                    estilo = gr.Dropdown(list(gen.STYLES), value="silueta", label="Estilo")
                    semilla = gr.Number(value=0, label="Semilla", precision=0)
                    use_llm = gr.Checkbox(True, label="Traducir / mejorar con Ollama")
                    with gr.Row():
                        web = gr.Checkbox(False, label="Nutrir con imágenes de internet (agente buscador)")
                        nrefs = gr.Slider(1, 6, 3, step=1, label="Variantes")
                    bgen = gr.Button("Generar", variant="primary")
                with gr.Tab("Imagen"):
                    img = gr.Image(type="filepath", label="Tu imagen", height=260)
                    itexto = gr.Textbox(label="Qué es / cómo se mueve (vacío = lo describe la IA de visión)")
                    with gr.Row():
                        iestilo = gr.Dropdown(list(gen.STYLES), value="silueta", label="Estilo")
                        imov = gr.Dropdown(list(gen.MOTIONS), value="flotar", label="Movimiento base")
                    ifuerza = gr.Slider(0, 0.9, 0.6, step=0.05, label="Fuerza IA (0 = sin IA, solo mueve tu imagen)")
                    with gr.Row():
                        isemilla = gr.Number(value=0, label="Semilla", precision=0)
                        illm = gr.Checkbox(True, label="Usar Ollama")
                    bimg = gr.Button("Animar", variant="primary")
                with gr.Tab("Vídeo / GIF"):
                    archivo = gr.File(type="filepath", label="Vídeo, GIF o imagen")
                    bvid = gr.Button("Pixelizar", variant="primary")
                with gr.Accordion("Pixelado", open=False):
                    res = gr.Slider(48, 192, 96, step=16, label="Resolución")
                    paleta = gr.Dropdown(list(px.PALETTES), value="gris", label="Paleta")
                    tonos = gr.Slider(1, 5, 3, step=1, label="Tonos de tinta")
                    grano = gr.Slider(0, 1, 0.35, label="Grano vivo")
                    motas = gr.Slider(0, 2, 0.6, label="Motas en el borde")
                    contraste = gr.Slider(0.5, 3, 1.3, label="Contraste")
                    umbral = gr.Slider(0.2, 0.8, 0.5, label="Umbral (bajo = más grueso)")
                    zoom = gr.Slider(0.6, 2.5, 1.0, label="Zoom")
                    ms = gr.Slider(40, 250, 80, step=10, label="ms por fotograma")
                    bre = gr.Button("Reajustar sin regenerar")
            with gr.Column(scale=1):
                gif = gr.Image(label="Resultado", type="filepath", height=400)
                variantes = gr.Gallery(label="Variantes (clic para elegir)", columns=6, height=130, allow_preview=False)
                info = gr.Textbox(label="Info", lines=3)
                files = gr.File(label="Archivos", file_count="multiple")
                refs = gr.Gallery(label="Referencias encontradas en internet", columns=6, height=130)
        pix = [res, paleta, tonos, grano, motas, contraste, umbral, zoom, ms]
        outs = [gif, files, info, variantes]
        box.submit(do_chat, [box, chatbot], [box, chatbot, gif, files, info, variantes, refs, *pix])
        breset.click(do_chat_reset, None, chatbot)
        bgen.click(do_text, [texto, estilo, semilla, use_llm, web, nrefs, *pix], outs + [refs])
        bimg.click(do_image, [img, itexto, iestilo, imov, ifuerza, isemilla, illm, *pix], outs)
        bvid.click(do_video, [archivo, *pix], outs)
        bre.click(do_retune, pix, outs)
        variantes.select(do_pick, pix, [gif, files, info])
        # vida de la sesión: al cerrar la pestaña se apaga el servidor
        demo.load(_hello, None, None, queue=False, show_progress="hidden")
        if hasattr(demo, "unload"):
            demo.unload(_bye)
        if hasattr(gr, "Timer"):
            gr.Timer(5).tick(_hello, None, None, queue=False, show_progress="hidden")
    return demo


def launch(inbrowser=True, port=None, auto_close=True):
    demo = build()
    if auto_close and os.environ.get("PIXELANIM_NO_AUTOCLOSE") != "1":
        threading.Thread(target=_watchdog, daemon=True).start()
        print("Al cerrar la pestaña del navegador, PixelAnim se cierra solo y libera el puerto.", flush=True)
    kw = {"inbrowser": inbrowser}
    if port:
        kw["server_port"] = port
    if "theme" not in inspect.signature(gr.Blocks.__init__).parameters:
        kw["theme"] = gr.themes.Monochrome()
    demo.queue(default_concurrency_limit=1).launch(**kw)


if __name__ == "__main__":
    launch()
