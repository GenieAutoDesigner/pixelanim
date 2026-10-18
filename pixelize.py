"""Pixelizador: fotogramas (PIL/arrays) -> sprite tramado estilo 1-bit/low-bit con ruido vivo.

Pipeline:
  1. recorte cuadrado + reducción por área a R×R
  2. oscuridad 0..1 (sujeto oscuro sobre fondo claro; auto-invierte si el fondo es oscuro)
  3. niveles globales del clip (no por fotograma -> sin parpadeo)
  4. limpieza de fondo + suavizado temporal circular
  5. bucle sin corte (fundido cruzado del final con el principio)
  6. tramado ordenado Bayer 8×8 + jitter por fotograma (el "grano" que se mueve)
  7. motas sueltas en el borde de la silueta (efecto desmoronado)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PALETTES = {
    # fondo, rampa de tinta (de claro a oscuro). El fondo no se dibuja en CSS.
    "gris":    ("#b9b8b2", ["#8a8984", "#4c4b48", "#161616"]),
    "fosforo": ("#0a130c", ["#1d5a2c", "#3fae5e", "#8dffa8"]),
    "ambar":   ("#170f05", ["#5c3a0b", "#a86d14", "#ffbd57"]),
    "papel":   ("#efe6d2", ["#c9a98a", "#8b5a3c", "#2a1a12"]),
    "sangre":  ("#0d0506", ["#4a0a10", "#a3141f", "#ff3b3b"]),
    "hielo":   ("#0b1018", ["#23405e", "#4f8fc0", "#cfeaff"]),
    "gameboy": ("#9bbc0f", ["#8bac0f", "#306230", "#0f380f"]),
}

_B2 = np.array([[0, 2], [3, 1]])
_B8 = _B2
for _ in range(2):
    _B8 = np.block([[4 * _B8, 4 * _B8 + 2], [4 * _B8 + 3, 4 * _B8 + 1]])
BAYER8 = (_B8 + 0.5) / 64.0


def _hash(shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _to_square(img: Image.Image, R: int, zoom: float = 1.0) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h) / zoom
    l, t = (w - s) / 2, (h - s) / 2
    img = img.crop((int(l), int(t), int(l + s), int(t + s)))
    img = img.resize((R, R), Image.Resampling.BOX if s >= R else Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return 1.0 - (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114)


def _blur3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    k = [(0, 0, 4), (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2), (-1, -1, 1), (1, 1, 1), (-1, 1, 1), (1, -1, 1)]
    out = np.zeros_like(a)
    H, W = a.shape
    for dy, dx, w in k:
        out += w * p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    return out / 16.0


def _dilate(m: np.ndarray, r: int = 2) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def clean_mask(m: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Quita motas y manchas pequeñas sueltas: conserva los componentes conexos grandes."""
    H, W = m.shape
    lab = np.zeros((H, W), np.int32)
    sizes = [0]
    cur = 0
    for y0 in range(H):
        for x0 in range(W):
            if m[y0, x0] and not lab[y0, x0]:
                cur += 1
                stack = [(y0, x0)]
                lab[y0, x0] = cur
                n = 0
                while stack:
                    y, x = stack.pop()
                    n += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = cur
                            stack.append((yy, xx))
                sizes.append(n)
    if cur == 0:
        return m
    sizes = np.array(sizes)
    keep = sizes >= max(4, min_frac * sizes[1:].max())
    keep[0] = False
    return keep[lab]


def autofit_box(frames, pad: float = 0.08):
    """Caja cuadrada (en píxeles de origen) que contiene al sujeto en todos los fotogramas."""
    W, H = frames[0].size
    S = 128
    lum = np.stack([np.asarray(f.convert("L").resize((S, S), Image.Resampling.BOX), np.float32) / 255 for f in frames])
    k = S // 16
    ring = np.concatenate([lum[:, k, k:-k].ravel(), lum[:, -k - 1, k:-k].ravel(), lum[:, k:-k, k].ravel(), lum[:, k:-k, -k - 1].ravel()])
    bg = float(np.median(ring))
    diff = np.abs(lum - bg).max(0)
    diff[:k] = diff[-k:] = 0; diff[:, :k] = diff[:, -k:] = 0
    m = clean_mask(diff > 0.25)
    if m.sum() < 20:
        return None
    ys, xs = np.nonzero(m)
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    cx, cy = (x0 + x1) / 2 / S * W, (y0 + y1) / 2 / S * H
    side = max(x1 - x0, y1 - y0) / S * max(W, H) * (1 + 2 * pad)
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def make_loop(frames: list[np.ndarray], fade: int) -> list[np.ndarray]:
    """Funde los últimos `fade` fotogramas sobre los primeros -> bucle continuo."""
    n = len(frames)
    if fade <= 0 or n < fade * 2 + 2:
        return frames
    out = [f.copy() for f in frames[: n - fade]]
    for i in range(fade):
        t = (i + 1) / (fade + 1)
        out[i] = frames[n - fade + i] * (1 - t) + frames[i] * t
    return out


def pixelize(
    frames: list[Image.Image],
    res: int = 96,
    levels: int = 3,
    grain: float = 0.35,
    specks: float = 0.5,
    contrast: float = 1.0,
    threshold: float = 0.5,
    smooth: float = 0.25,
    loop_fade: int = 3,
    zoom: float = 1.0,
    invert: str | bool = "auto",
    seed: int = 0,
    autofit: bool = True,
) -> list[np.ndarray]:
    """Devuelve una lista de arrays int8 R×R con índice de tinta (0 = vacío, 1..levels)."""
    if autofit:
        box = autofit_box(frames)
        if box:
            # recorte (puede salirse del lienzo: se rellena con el color del borde interior)
            W, H = frames[0].size
            src = []
            for f in frames:
                f = f.convert("RGB")
                k = max(2, W // 16)
                edge = tuple(int(c) for c in np.median(np.asarray(f)[k:-k:8, k], axis=0))
                canvas = Image.new("RGB", (int(box[2] - box[0]), int(box[3] - box[1])), edge)
                canvas.paste(f.crop((max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3])))),
                             (max(0, -int(box[0])), max(0, -int(box[1]))))
                src.append(canvas)
            frames = src
    d = [_to_square(f, res, zoom) for f in frames]          # oscuridad 0..1
    lum = np.stack([1.0 - a for a in d])                    # luminancia
    R0 = res
    k = max(2, R0 // 16)                                    # anillo interior (evita marcos negros del borde)
    ring = np.concatenate([lum[:, k, k:-k].ravel(), lum[:, -k - 1, k:-k].ravel(),
                           lum[:, k:-k, k].ravel(), lum[:, k:-k, -k - 1].ravel()])
    bgl = float(np.median(ring))
    if invert == "auto":
        invert = bgl < 0.35                                 # fondo oscuro -> sujeto claro
    if invert:
        v = np.clip((lum - bgl) / max(1e-3, 1 - bgl), 0, 1)
    else:
        v = np.clip((bgl - lum) / max(1e-3, bgl), 0, 1)     # cuánto más oscuro que el fondo
    # recortar lo que queda fuera del anillo (marcos)
    v[:, :k, :] = 0; v[:, -k:, :] = 0; v[:, :, :k] = 0; v[:, :, -k:] = 0
    hi = max(float(np.percentile(v, 99.5)), 0.05)
    v = [np.clip(a / hi, 0, 1) for a in v]

    # contraste alrededor del umbral
    v = [np.clip((a - threshold) * contrast + 0.5, 0, 1) for a in v]

    # suavizado temporal circular (quita parpadeo de la IA, mantiene forma)
    if smooth > 0 and len(v) > 2:
        n = len(v)
        v = [(1 - smooth) * v[i] + smooth * 0.5 * (v[i - 1] + v[(i + 1) % n]) for i in range(n)]

    # quitar motas sueltas de la IA (puntos aislados que ensucian el fondo)
    v = [a * clean_mask(a > 0.3, 0.01) + a * (a <= 0.3) * _dilate(clean_mask(a > 0.3, 0.01), 2) for a in v]

    v = make_loop(v, loop_fade)
    v = [_blur3(a) * 0.5 + a * 0.5 for a in v]

    R = res
    tile = np.tile(BAYER8, (R // 8 + 1, R // 8 + 1))[:R, :R].astype(np.float32)
    out = []
    for fi, a in enumerate(v):
        jit = (_hash((R, R), seed * 1000 + fi) - 0.5) * grain
        t = np.clip(tile + jit, 0, 1)
        # cuantización ordenada a `levels` tonos de tinta (0 = vacío)
        q = a * levels
        base = np.floor(q)
        idx = (base + ((q - base) > t)).astype(np.int8)
        idx = np.clip(idx, 0, levels)
        # zonas casi sólidas: tinta máxima limpia
        idx[a > 0.93] = levels
        # motas sueltas alrededor del borde
        if specks > 0:
            body = a > 0.45
            ring = _dilate(body, 2) & ~body & (a < 0.3)
            r = _hash((R, R), seed * 7919 + fi * 31 + 1)
            idx[ring & (r < 0.06 * specks)] = max(1, levels - 1)
            idx[ring & (r > 1 - 0.02 * specks)] = levels
        out.append(idx)
    return out


# ---------------------------------------------------------------- exportadores

def _ramp(palette: str, levels: int):
    bg, ink = PALETTES[palette]
    if levels == 1:
        cols = [ink[-1]]
    elif levels == 2:
        cols = [ink[1], ink[2]]
    else:
        # interpolar la rampa a `levels` tonos
        def h2rgb(h):
            return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
        stops = [h2rgb(c) for c in ink]
        cols = []
        for k in range(levels):
            x = k / (levels - 1) * (len(stops) - 1)
            i = min(int(x), len(stops) - 2)
            c = stops[i] * (1 - (x - i)) + stops[i + 1] * (x - i)
            cols.append("#%02x%02x%02x" % tuple(int(round(z)) for z in c))
    return bg, cols


def _short(h: str) -> str:
    if h[1] == h[2] and h[3] == h[4] and h[5] == h[6]:
        return "#" + h[1] + h[3] + h[5]
    return h


def to_rgb_frames(idx_frames, palette="gris", levels=3, scale=4, transparent=False):
    bg, cols = _ramp(palette, levels)
    lut = np.zeros((levels + 1, 4), np.uint8)
    lut[0] = [int(bg[i:i + 2], 16) for i in (1, 3, 5)] + [0 if transparent else 255]
    for k, c in enumerate(cols):
        lut[k + 1] = [int(c[i:i + 2], 16) for i in (1, 3, 5)] + [255]
    imgs = []
    for f in idx_frames:
        a = lut[f]
        im = Image.fromarray(a, "RGBA")
        if scale > 1:
            im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
        imgs.append(im)
    return imgs


def export_gif(idx_frames, path, palette="gris", levels=3, scale=4, ms=80):
    imgs = [i.convert("RGB") for i in to_rgb_frames(idx_frames, palette, levels, scale)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=False, disposal=1)


def export_sheet(idx_frames, path, palette="gris", levels=3):
    imgs = to_rgb_frames(idx_frames, palette, levels, 1, transparent=True)
    R = imgs[0].width
    sheet = Image.new("RGBA", (R * len(imgs), R), (0, 0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * R, 0))
    sheet.save(path)


def export_css(idx_frames, name="sprite", palette="gris", levels=3, px=4, ms=80) -> str:
    _, cols = _ramp(palette, levels)
    cols = [_short(c) for c in cols]
    shadows = []
    for f in idx_frames:
        ys, xs = np.nonzero(f)
        parts = [f"{(x + 1) * px}px {(y + 1) * px}px {cols[f[y, x] - 1]}" for y, x in zip(ys, xs)]
        shadows.append(",".join(parts) if parts else "none")
    n = len(shadows)
    css = (f".{name}{{width:{px}px;height:{px}px;margin:-{px}px 0 0 -{px}px;box-shadow:{shadows[0]};"
           f"animation:{name} {n * ms}ms steps(1) infinite}}\n@keyframes {name}{{\n")
    css += "\n".join(f"{i / n * 100:.3f}%{{box-shadow:{s}}}" for i, s in enumerate(shadows))
    css += f"\n100%{{box-shadow:{shadows[-1]}}}\n}}\n"
    return css


def export_css_demo(idx_frames, path, name="sprite", palette="gris", levels=3, px=4, ms=80):
    R = idx_frames[0].shape[0]
    bg, _ = _ramp(palette, levels)
    css = export_css(idx_frames, name, palette, levels, px, ms)
    html = f"""<!doctype html><meta charset="utf-8"><title>{name}</title>
<style>html,body{{height:100%;margin:0;background:{bg};display:grid;place-items:center}}
.box{{position:relative;width:{R * px}px;height:{R * px}px}}.box>div{{position:absolute}}
{css}</style><div class="box"><div class="{name}"></div></div>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return css


def _pack(idx_frames) -> str:
    """Fotogramas -> base64 de bytes (1 byte/píxel, RLE simple). Para el reproductor canvas."""
    import base64
    out = bytearray()
    for f in idx_frames:
        flat = f.ravel()
        i = 0
        while i < flat.size:
            v = int(flat[i])
            j = i
            while j < flat.size and flat[j] == v and j - i < 255:
                j += 1
            out += bytes([v, j - i])
            i = j
    return base64.b64encode(bytes(out)).decode()


def export_wallpaper(idx_frames, path, palette="gris", levels=3, ms=80, size=0.55, drift=True, title="Sprite"):
    """HTML a pantalla completa (Lively Wallpaper / Wallpaper Engine / navegador). Canvas, ligero."""
    R = idx_frames[0].shape[0]
    bg, cols = _ramp(palette, levels)
    data = _pack(idx_frames)
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{{margin:0;height:100%;background:{bg};overflow:hidden}}canvas{{position:absolute;image-rendering:pixelated}}</style>
<canvas id="c"></canvas><script>
const R={R},N={len(idx_frames)},MS={ms},SIZE={size},DRIFT={str(drift).lower()},COLS={cols!r};
const raw=Uint8Array.from(atob("{data}"),c=>c.charCodeAt(0));const fr=[];let p=0;
for(let f=0;f<N;f++){{const a=new Uint8Array(R*R);let i=0;while(i<R*R){{const v=raw[p++],n=raw[p++];a.fill(v,i,i+n);i+=n}}fr.push(a)}}
const c=document.getElementById('c'),x=c.getContext('2d');c.width=c.height=R;
const img=x.createImageData(R,R),rgb=COLS.map(h=>[1,3,5].map(i=>parseInt(h.substr(i,2),16)));
function draw(f){{const a=fr[f],d=img.data;for(let i=0;i<R*R;i++){{const v=a[i];if(v){{const q=rgb[v-1];d[i*4]=q[0];d[i*4+1]=q[1];d[i*4+2]=q[2];d[i*4+3]=255}}else d[i*4+3]=0}}x.putImageData(img,0,0)}}
function fit(){{const s=Math.floor(Math.min(innerWidth,innerHeight)*SIZE/R)||1;c.style.width=c.style.height=R*s+'px';return R*s}}
let side=fit();addEventListener('resize',()=>side=fit());
let f=0,last=0,t0=performance.now();
function tick(t){{if(t-last>=MS){{draw(f);f=(f+1)%N;last=t}}
 const k=(t-t0)/1000,dx=DRIFT?Math.sin(k*.23)*innerWidth*.18:0,dy=DRIFT?Math.sin(k*.31+1)*innerHeight*.12:0;
 c.style.left=(innerWidth-side)/2+dx+'px';c.style.top=(innerHeight-side)/2+dy+'px';requestAnimationFrame(tick)}}
requestAnimationFrame(tick);
</script>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def export_all(idx_frames, outdir, name, palette="gris", levels=3, ms=80, px=None):
    import os
    os.makedirs(outdir, exist_ok=True)
    R = idx_frames[0].shape[0]
    px = px or max(2, round(384 / R))
    export_gif(idx_frames, os.path.join(outdir, f"{name}.gif"), palette, levels, px, ms)
    export_sheet(idx_frames, os.path.join(outdir, f"{name}_sheet.png"), palette, levels)
    css = export_css_demo(idx_frames, os.path.join(outdir, f"{name}_css.html"), name, palette, levels, px, ms)
    with open(os.path.join(outdir, f"{name}.css"), "w", encoding="utf-8") as fh:
        fh.write(css)
    export_wallpaper(idx_frames, os.path.join(outdir, f"{name}_wallpaper.html"), palette, levels, ms, title=name)
    return {"css_kb": round(len(css) / 1024), "frames": len(idx_frames), "res": R}
