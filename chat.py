"""Asistente por chat: convierte lo que pides en lenguaje natural en parámetros y acciones.

Con Ollama abierto usa tu LLM local. Sin Ollama, entiende órdenes básicas por palabras clave.
"""
from __future__ import annotations

import copy
import json
import re

import llm
import pixelize as px

STYLES = ("silueta", "tinta", "sombra", "detalle")
MOTIONS = ("flotar", "respirar", "balanceo", "viento", "aleteo", "ninguno")

DEFAULTS = {
    "tema": "", "estilo": "silueta", "movimiento": "flotar", "fuerza": 0.6, "semilla": 0,
    "res": 96, "paleta": "gris", "tonos": 3, "grano": 0.35, "motas": 0.6,
    "contraste": 1.3, "umbral": 0.5, "zoom": 1.0, "ms": 80, "web": False,
}
LIMITS = {
    "fuerza": (0.0, 0.9), "semilla": (0, 10**6), "res": (48, 192), "tonos": (1, 5), "grano": (0, 1),
    "motas": (0, 2), "contraste": (0.5, 3), "umbral": (0.2, 0.8), "zoom": (0.6, 2.5), "ms": (40, 250),
}
PIXEL_KEYS = {"res", "paleta", "tonos", "grano", "motas", "contraste", "umbral", "zoom", "ms"}

SYSTEM = f"""Eres el asistente de PixelAnim, un generador de sprites animados pixelados y tramados (estilo 1-bit retro).
El usuario te dice en español lo que quiere. Tú respondes SOLO con un objeto JSON:
{{"respuesta": "frase corta en español para el usuario",
  "accion": "generar" | "reajustar" | "nada",
  "cambios": {{ parámetros que cambian }} }}

Parámetros (usa solo estos nombres y rangos):
- tema: descripción del sujeto y su movimiento, SIEMPRE TRADUCIDA AL INGLÉS, para Stable Diffusion (ej. "a bat flapping its wings, front view"). Un solo sujeto.
- estilo: {list(STYLES)}  (silueta = forma negra plana; tinta = ilustración a pincel; sombra = humo/espectral; detalle = ilustración detallada)
- movimiento: {list(MOTIONS)}  (solo se usa cuando hay imagen de entrada)
- fuerza: 0.0-0.9  (solo con imagen: 0 = sin IA, solo mueve la imagen; 0.4 fiel; 0.8 la IA reinterpreta)
- semilla: entero (cambiarla da otra versión)
- res: 48-192 (píxeles del sprite; 64 = muy retro, 96 normal, 128+ detallado)
- paleta: {list(px.PALETTES)}
- tonos: 1-5 (1 = 1-bit puro)
- grano: 0-1 (ruido que cambia en cada fotograma)
- motas: 0-2 (píxeles sueltos que se desprenden del borde)
- contraste: 0.5-3 ; umbral: 0.2-0.8 (umbral bajo = silueta más gruesa)
- zoom: 0.6-2.5 ; ms: 40-250 (ms por fotograma; menos = más rápido)
- web: true/false (true = antes de generar, un agente busca fotos de referencia del tema en internet y anima las mejores;
  más fiel a la realidad. Ponlo a true si pide "busca en internet", "con referencias", "como en la realidad", "fotos reales")

Reglas:
- "generar" si cambia tema, estilo, movimiento, fuerza, semilla o web, o si el usuario adjunta una imagen nueva.
- "reajustar" si solo cambian parámetros de pixelado (res, paleta, tonos, grano, motas, contraste, umbral, zoom, ms).
- "nada" si solo pregunta algo.
- Si pide "otra versión" o no le gusta el resultado, cambia la semilla y genera.
- Si adjunta imagen y no describe nada, deja tema vacío "".
- No inventes parámetros. Respuesta breve."""


class Agent:
    def __init__(self):
        self.params = copy.deepcopy(DEFAULTS)
        self.history: list[dict] = []
        self.has_result = False

    # ------------------------------------------------------------------ público
    def step(self, msg: str, image_attached: bool = False, image_available: bool = False):
        """-> (respuesta, accion, params)"""
        msg = (msg or "").strip()
        state = {k: self.params[k] for k in self.params}
        user = msg + ("\n[El usuario ha adjuntado una imagen]" if image_attached else "")
        user += f"\n[Estado actual: {json.dumps(state, ensure_ascii=False)}; ya hay resultado: {self.has_result}]"
        # 1) órdenes claras (paleta, tamaño, velocidad, «otra versión», internet…): reglas deterministas, instantáneas.
        # 2) lo demás (tema nuevo, preguntas, frases libres): LLM local; si no hay o falla, reglas.
        rules = self._rules(msg, image_attached)
        explicit = set(rules["cambios"]) - {"tema"}
        if explicit or rules["accion"] == "reajustar":
            out, via = rules, "reglas"
        else:
            out = llm.chat_json([{"role": "system", "content": SYSTEM}, *self.history[-8:],
                                 {"role": "user", "content": user}])
            if not isinstance(out, dict) or out.get("accion") not in ("generar", "reajustar", "nada"):
                out, via = rules, "reglas"
            else:
                via = "llm"
                ch = out.get("cambios") if isinstance(out.get("cambios"), dict) else {}
                out["cambios"] = ch
                # red de seguridad para LLM pequeños: si el usuario describe algo nuevo y el LLM no lo recoge
                if rules["accion"] == "generar" and "tema" in rules["cambios"]:
                    if out["accion"] == "nada" or (out["accion"] == "generar" and not str(ch.get("tema", "")).strip()):
                        ch["tema"] = rules["cambios"]["tema"]
                        out["accion"] = "generar"
        changes = self._validate(out.get("cambios") or {})
        if via == "llm" and "semilla" in changes and not re.search(r"(otra|otro|version|versión|cambia|repite|semilla)",
                                                                    msg.lower()):
            changes.pop("semilla")                      # los LLM pequeños inventan semillas
        if image_attached:
            changes["web"] = False                      # con imagen propia no se busca en internet
            if not msg:
                changes["tema"] = ""                    # imagen sin texto: que la describa la IA de visión
        action = out.get("accion", "nada")
        if image_attached and action == "nada":
            action = "generar"
        if action == "reajustar" and not self.has_result:
            action = "generar" if (self.params["tema"] or image_attached or image_available) else "nada"
        if action == "reajustar" and set(changes) - PIXEL_KEYS:
            action = "generar"
        self.params.update(changes)
        if action == "generar" and not (self.params["tema"] or image_attached or image_available):
            action = "nada"
            out["respuesta"] = "Dime qué quieres que anime (o adjunta una imagen)."
        reply = out.get("respuesta") or "Hecho."
        self.history += [{"role": "user", "content": msg}, {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}]
        return reply, action, dict(self.params), via

    # ------------------------------------------------------------------ interno
    def _validate(self, ch: dict) -> dict:
        ok = {}
        for k, v in ch.items():
            if k not in DEFAULTS:
                continue
            try:
                if k in LIMITS:
                    lo, hi = LIMITS[k]
                    v = type(DEFAULTS[k])(float(v))
                    v = min(max(v, lo), hi)
                elif k == "estilo" and v not in STYLES:
                    continue
                elif k == "movimiento" and v not in MOTIONS:
                    continue
                elif k == "paleta":
                    v = str(v).lower().replace("ó", "o").replace("á", "a")
                    if v not in px.PALETTES:
                        continue
                elif k == "tema":
                    v = str(v)[:200]
                elif k == "web":
                    v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "si", "sí", "yes")
                ok[k] = v
            except (TypeError, ValueError):
                continue
        return ok

    def _rules(self, msg: str, image: bool) -> dict:
        t = msg.lower()
        t2 = t.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        ch, pix_only = {}, True
        p = self.params
        for pal in px.PALETTES:
            if pal in t2:
                ch["paleta"] = pal
        aliases = {"verde": "fosforo", "naranja": "ambar", "rojo": "sangre", "azul": "hielo", "crema": "papel", "gris": "gris"}
        for w, pal in aliases.items():
            if re.search(rf"\b{w}", t2):
                ch["paleta"] = pal
        for s in STYLES:
            if s in t2:
                ch["estilo"] = s; pix_only = False
        stems = {"flot": "flotar", "respir": "respirar", "balance": "balanceo", "viento": "viento", "ondul": "viento",
                 "alete": "aleteo", "quiet": "ninguno", "sin movimiento": "ninguno"}
        for st, m in stems.items():
            if st in t2:
                ch["movimiento"] = m; pix_only = False
        menos = "menos" in t2
        if re.search(r"(mas grande|mas detalle|mas resolucion)", t2): ch["res"] = p["res"] + 32
        if re.search(r"(mas pequeno|menos detalle|mas retro|menos resolucion)", t2): ch["res"] = p["res"] - 32
        if "rapid" in t2: ch["ms"] = p["ms"] - 20
        if "lent" in t2: ch["ms"] = p["ms"] + 30
        if re.search(r"(grano|ruido)", t2): ch["grano"] = p["grano"] + (-0.15 if menos else 0.15)
        if "mota" in t2: ch["motas"] = p["motas"] + (-0.4 if menos else 0.4)
        if "grues" in t2: ch["umbral"] = p["umbral"] - 0.07
        if "fino" in t2 or "delgad" in t2: ch["umbral"] = p["umbral"] + 0.07
        if "contraste" in t2: ch["contraste"] = p["contraste"] + (-0.3 if menos else 0.3)
        if "1 bit" in t2 or "1-bit" in t2: ch["tonos"] = 1
        if "zoom" in t2 or "acerca" in t2: ch["zoom"] = p["zoom"] + 0.2
        if "aleja" in t2: ch["zoom"] = p["zoom"] - 0.2
        m = re.search(r"\b(48|64|80|96|112|128|144|160|176|192)\b", t2)
        if m: ch["res"] = int(m.group(1))
        if re.search(r"(otra|otro|no me gusta|repite|de nuevo|cambia)", t2) and not ch:
            ch["semilla"] = p["semilla"] + 1; pix_only = False
        if "sin ia" in t2: ch["fuerza"] = 0.0; pix_only = False
        if re.search(r"(internet|busca|referencia|fotos reales|de verdad|realista)", t2):
            neg = bool(re.search(r"(sin internet|no busques|sin referencia|no uses internet)", t2))
            ch["web"] = not neg; pix_only = False
            rest = re.sub(r"(busca(me)?|buscando|en internet|por internet|internet|con referencias?|referencias?|"
                          r"fotos reales|y|de|en|una?|el|la|los|las|imagenes|imágenes|fotos)\b", " ", t, flags=re.I)
            rest = re.sub(r"\s+", " ", rest).strip(" ,.")
            if len(rest) > 2 and not neg:
                ch["tema"] = rest
        if re.fullmatch(r"\W*(hola|buenas|gracias|vale|ok|que tal|como va)\b.*", t2) or t2.endswith("?"):
            return {"respuesta": "Dime qué animar (p. ej. «un cuervo volando») o adjunta una imagen. "
                                 "Luego puedes pedir cambios: «en verde», «más grande», «más rápido», «otra versión».",
                    "accion": "nada", "cambios": {}}
        if image and not ch:
            ch["tema"] = msg  # vacío = autodescripción de la imagen
        if not ch and not image and len(t.split()) >= 1:
            ch["tema"] = msg; pix_only = False      # se traduce luego con enhance_prompt si hay LLM
        if image:
            pix_only = False
        accion = "reajustar" if (ch and pix_only) else ("generar" if ch or image else "nada")
        names = {"paleta": "paleta", "res": "resolución", "ms": "ms/fotograma", "grano": "grano", "motas": "motas",
                 "umbral": "umbral", "contraste": "contraste", "tonos": "tonos", "zoom": "zoom", "estilo": "estilo",
                 "movimiento": "movimiento", "fuerza": "fuerza IA", "semilla": "semilla"}
        parts = []
        for k, v in ch.items():
            if k == "tema":
                parts.append(f"tema «{v}»")
            elif k == "web":
                parts.append("con referencias de internet" if v else "sin internet")
            elif k in names:
                parts.append(f"{names[k]} {round(v, 2) if isinstance(v, float) else v}")
        if image and "tema" not in ch:
            parts.insert(0, "animo tu imagen")
        reply = ("Vale: " + ", ".join(parts) + ".") if parts else "No te he entendido."
        return {"respuesta": reply, "accion": accion, "cambios": ch}
