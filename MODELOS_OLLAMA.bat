@echo off
setlocal
REM Descarga los modelos de Ollama que usa PixelAnim. Doble clic y listo.
set "OLL="
for %%P in ("%LOCALAPPDATA%\Programs\Ollama\ollama.exe" "%ProgramFiles%\Ollama\ollama.exe" "%ProgramFiles(x86)%\Ollama\ollama.exe" "%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe") do (
  if not defined OLL if exist %%P set "OLL=%%~P"
)
if not defined OLL (
  for /f "delims=" %%W in ('where ollama 2^>nul') do if not defined OLL set "OLL=%%W"
)
if not defined OLL (
  echo [ERROR] No encuentro ollama.exe. Instala Ollama desde https://ollama.com/download/windows
  pause & exit /b 1
)
echo Ollama: "%OLL%"

REM Arrancar el servidor si no esta en marcha
"%OLL%" list >nul 2>nul
if errorlevel 1 (
  echo Arrancando Ollama...
  start "" /min "%OLL%" serve
  timeout /t 5 /nobreak >nul
)

echo.
echo Elige que modelos descargar:
echo   1 = Normal  (qwen2.5:7b + llava:7b, ~9.4 GB)  - GPU de 8 GB o mas
echo   2 = Ligero  (llama3.2:3b + moondream, ~3.7 GB) - GPU de 6 GB o menos / portatil
choice /c 12 /n /m "Opcion [1/2]: "
if errorlevel 2 goto :ligero
echo.
echo === qwen2.5:7b (chat, ~4.7 GB) ===
"%OLL%" pull qwen2.5:7b || goto :fail
echo.
echo === llava:7b (vision, ~4.7 GB) ===
"%OLL%" pull llava:7b || goto :fail
goto :fin
:ligero
echo.
echo === llama3.2:3b (chat, ~2 GB) ===
"%OLL%" pull llama3.2:3b || goto :fail
echo.
echo === moondream (vision, ~1.7 GB) ===
"%OLL%" pull moondream || goto :fail
:fin
echo.
"%OLL%" list
echo.
echo Modelos listos. Abre PixelAnim con INICIAR.bat
pause
exit /b 0

:fail
echo [ERROR] Fallo la descarga. Revisa la conexion y vuelve a ejecutar este archivo (continua donde se quedo).
pause
exit /b 1
