# PixelAnim

Generador local de sprites animados tramados. Partes de un texto, de una imagen o de fotos que busca en internet, una IA local hace la animación y el programa la pixela.

Exporta:

- `.css`: un único `div` con `box-shadow`, sin imágenes.
- `.gif`
- `_sheet.png`: sprite sheet.
- `_wallpaper.html`: fondo de pantalla en canvas, ligero.

## Instalación (Windows)

1. No necesitas instalar Python: `INICIAR.bat` usa uno portable dentro de la carpeta. Con GPU NVIDIA (6 GB de VRAM o más) genera en segundos; sin GPU va por CPU y tarda minutos.
2. Doble clic en `INICIAR.bat`. Instala Python portable y todo lo necesario dentro de la carpeta, pasa una comprobación automática y abre la interfaz.
3. Las siguientes veces, `INICIAR.bat` abre directamente.
4. La primera generación descarga unos 2,5 GB de modelos (una sola vez). La primera búsqueda en internet descarga unos 0,8 GB más (CLIP y el recortador de fondos).

Comprobación completa con IA: `python\python.exe pixelanim.py probar --ia`.

## Formas de usarlo (pestañas)

| Pestaña | Qué hace |
|---|---|
| **Chat** | Le pides lo que quieres en lenguaje natural y puedes adjuntar una imagen con el clip. Después ajustas hablando: «en verde», «más grande», «más rápido», «otra versión», «estilo tinta», «que se balancee», «sin IA», «busca en internet…». |
| **Texto** | Escribes el texto y obtienes la animación. Si marcas **Nutrir con imágenes de internet**, el agente buscador encuentra fotos reales del tema y anima las mejores (salen varias variantes para elegir). |
| **Imagen** | Tu imagen se convierte en animación. Hay un movimiento base (flotar, respirar, balanceo, viento o aleteo) y luego la IA la reinterpreta con la *fuerza* que elijas. Con fuerza 0 no usa IA: solo mueve y pixela tu imagen, en un segundo. |
| **Vídeo / GIF** | Pixela un clip que ya tengas. |

Al hacer clic en una **variante** pasa a ser el resultado principal. «Reajustar sin regenerar» vuelve a pixelar todas las variantes con los nuevos ajustes.

## Agente buscador de imágenes (internet)

`buscador.py`, también disponible con `python pixelanim.py refs "búho volando"`:

1. **Consultas.** El LLM local escribe 4 búsquedas en inglés: silueta, ilustración en blanco y negro, fondo blanco y pose. Sin Ollama usa plantillas.
2. **Fuentes en paralelo**, sin cuentas ni claves:
   - DuckDuckGo Imágenes y Bing Imágenes, que indexan toda la web.
   - Las mismas búsquedas restringidas a **Pinterest, DeviantArt, Reddit y ArtStation** (`site:`).
   - Openverse (Flickr y otros repositorios con licencia CC).
   - Wikimedia Commons.

   Las redes con inicio de sesión obligatorio (Instagram, TikTok, X) no se raspan directamente: llegan sus imágenes públicas que tienen indexadas los buscadores.
3. **Descarga.** Hasta 60 imágenes en paralelo, con tiempo y tamaño máximos. Se descartan las que no son imagen, las muy pequeñas y las casi duplicadas.
4. **Puntuación**, con un 65 % de relevancia y un 35 % de pixelabilidad:
   - Relevancia: **CLIP** local mide si la foto es de verdad lo que pides y no un logo, un texto o un collage.
   - Pixelabilidad: fondo liso, contraste y sujeto centrado.
5. **Recorte.** **rembg** separa el sujeto y lo pone sobre fondo blanco.
6. **Animación.** Las mejores pasan al modo imagen, con un movimiento elegido según el tema (las aves aletean, los fantasmas flotan…).

Todo corre en un proceso aparte que, al terminar, libera la memoria antes de cargar la IA de vídeo. Las referencias se guardan en `salidas/web-…/referencias/`, junto con `referencias.json`, que recoge el origen y la página de cada una.

> Uso personal: las imágenes de terceros tienen sus derechos. Las de Openverse y Commons son CC: revisa su licencia en `referencias.json` si vas a publicar algo.

## Ollama (opcional, recomendado)

Con Ollama abierto:
- El chat entiende frases libres.
- Las búsquedas y los prompts se escriben en inglés.
- Si tienes un modelo de visión, se describen solas las imágenes sin texto.

Las órdenes claras («en verde», «64», «más rápido», «otra versión», «busca en internet») se resuelven al instante con reglas, sin LLM.

```
ollama pull qwen2.5:7b      # chat (si ya tienes llama3.1:8b, gemma o mistral, se usan esos)
ollama pull llava:7b        # visión (o moondream, más ligero)
```

Cada llamada descarga el modelo de Ollama de la VRAM al terminar (`keep_alive=0`), para que no compita con la IA de vídeo. Si prefieres que se quede cargado: `set PIXELANIM_OLLAMA_KEEP=5m`.

## Uso por consola

```
GENERAR.bat "murciélago volando" --res 96 --paleta gris --semillas 1 2 3
GENERAR.bat "búho volando" --web --variantes 3
python pixelanim.py gen "ghost floating" --estilo sombra --res 128 --tonos 4
python pixelanim.py imagen foto.png --movimiento aleteo --fuerza 0.6 --semillas 1 2 3
python pixelanim.py imagen logo.png --fuerza 0 --movimiento respirar      (sin IA)
python pixelanim.py refs "zorro corriendo" -n 8                           (solo buscar referencias)
python pixelanim.py video clip.mp4 --res 128 --paleta fosforo
python pixelanim.py probar [--ia]
```

Usa siempre el Python portable: `python\python.exe pixelanim.py …` (GENERAR.bat ya lo hace).

**Consejos para imágenes:** funciona mejor con un sujeto claro sobre un fondo liso. El sujeto se recorta y centra solo; aun así, los marcos dibujados se notan. Con fuerza 0.4–0.5 se respeta tu dibujo; con 0.7–0.8 la IA se toma más libertades y se mueve más.

Los resultados se guardan en `salidas/<nombre>/`. Los fotogramas originales de la IA quedan en `ia/`, para volver a pixelar sin regenerar.

## Cómo funciona

1. **IA**: AnimateDiff-Lightning (ByteDance, 4 pasos) sobre DreamShaper 8 (SD 1.5). Genera 16 fotogramas de 512×512 con movimiento coherente. A partir de una imagen usa vídeo a vídeo sobre un movimiento base periódico.
2. **Estilos**: `silueta`, `tinta`, `sombra` y `detalle`. Todos fuerzan un sujeto oscuro centrado sobre fondo blanco, que es lo que mejor se pixela.
3. **Pixelado** (`pixelize.py`):
   - Encuadre automático del sujeto y limpieza de motas sueltas.
   - Niveles globales del clip, para que no haya parpadeo.
   - Suavizado temporal circular y bucle sin corte por fundido cruzado.
   - Tramado Bayer 8×8 con jitter distinto en cada fotograma (el grano vivo).
   - Motas en el borde.

## Ajustes clave

| Parámetro | Efecto |
|---|---|
| `--res` | 64 = estilo del vídeo original · 96–128 = más detalle · 160+ = CSS muy pesado |
| `--tonos` | 1 = 1-bit puro · 3 = recomendado · 5 = casi degradado |
| `--grano` | 0 = trama estática · 0.35 = vivo · 0.8 = chisporroteo |
| `--motas` | píxeles que se desprenden del borde |
| `--contraste` / `--umbral` | grosor de la silueta |
| `--zoom` | acercar si el sujeto sale pequeño |

## Robustez

- Solo se hace una generación a la vez, aunque pulses en varias pestañas. Las demás esperan en cola.
- Si la GPU se queda sin memoria, se pasa solo a modo de descarga a RAM y se reintenta una vez.
- Con 6–7 GB de VRAM o menos, ese modo se activa desde el principio. Si aun así falta memoria: `set PIXELANIM_SIZE=384` antes de `INICIAR.bat` (algo menos de calidad).
- Cualquier error se muestra en la interfaz o en el chat con un mensaje claro. La aplicación no se cuelga.
- Búsquedas y descargas con tiempos máximos. Si una fuente falla o bloquea, se sigue con las demás.

## Límites

- El CSS pesa unos 0,1 KB por píxel pintado y fotograma: un sprite de 96×96 con 13 fotogramas ocupa entre 200 y 700 KB. Para fondos de pantalla usa `_wallpaper.html`, que pesa unas 20 veces menos.
- SD 1.5 a veces deforma la anatomía o mete fondo. En ese caso cambia la semilla, o activa internet para partir de fotos reales.
- Fondo de pantalla: en Lively Wallpaper, «Añadir» → selecciona `_wallpaper.html`.
