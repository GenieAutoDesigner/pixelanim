@echo off
REM Uso: GENERAR.bat "murcielago volando" [--web] [--res 96] [--paleta gris] [--semillas 1 2 3]
cd /d "%~dp0"
set "HF_HOME=%~dp0modelos\huggingface"
set "HF_HUB_DISABLE_XET=1"
set "U2NET_HOME=%~dp0modelos\u2net"
if not exist "%~dp0python\python.exe" (echo Primero abre INICIAR.bat & pause & exit /b 1)
"%~dp0python\python.exe" "%~dp0pixelanim.py" gen %*
