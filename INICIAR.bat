@echo off
setlocal EnableExtensions
title PixelAnim
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%python\python.exe"
set "HF_HOME=%ROOT%modelos\huggingface"
set "HF_HUB_DISABLE_XET=1"
set "U2NET_HOME=%ROOT%modelos\u2net"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo.
echo  ===============================
echo   PixelAnim  (version portable)
echo  ===============================
echo  Todo se guarda dentro de esta carpeta: Python, librerias y modelos.
echo.

if exist "%ROOT%.instalado" goto :lanzar

where curl >nul 2>nul || (echo [ERROR] Falta curl.exe ^(viene con Windows 10/11^). & pause & exit /b 1)
where tar  >nul 2>nul || (echo [ERROR] Falta tar.exe ^(viene con Windows 10/11^). & pause & exit /b 1)

REM ---------- 1. Python portable ----------
if not exist "%PY%" (
  echo [1/4] Descargando Python 3.12 portable...
  if not exist "%ROOT%python" mkdir "%ROOT%python"
  curl -L --fail --retry 3 -o "%ROOT%python\py.zip" https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip || goto :neterr
  tar -xf "%ROOT%python\py.zip" -C "%ROOT%python" || goto :neterr
  del "%ROOT%python\py.zip"
  (
    echo python312.zip
    echo .
    echo ..
    echo import site
  ) > "%ROOT%python\python312._pth"
)
REM ---------- 2. pip ----------
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  echo [2/4] Instalando pip...
  curl -L --fail --retry 3 -o "%ROOT%python\get-pip.py" https://bootstrap.pypa.io/get-pip.py || goto :neterr
  "%PY%" "%ROOT%python\get-pip.py" --no-warn-script-location || goto :piperr
  del "%ROOT%python\get-pip.py"
)
REM ---------- 3. PyTorch ----------
"%PY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
  nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo [3/4] GPU NVIDIA detectada: instalando PyTorch con CUDA ^(~3 GB, paciencia^)...
    "%PY%" -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cu128 || goto :piperr
  ) else (
    echo [3/4] Sin GPU NVIDIA: instalando PyTorch para CPU ^(ira lento^)...
    "%PY%" -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu || goto :piperr
  )
)
REM ---------- 4. Resto ----------
echo [4/4] Instalando el resto de librerias...
"%PY%" -m pip install --no-warn-script-location -r "%ROOT%requirements.txt" || goto :piperr

echo.
echo Comprobando la instalacion...
"%PY%" "%ROOT%pixelanim.py" probar
if errorlevel 1 (
  echo.
  echo [AVISO] La comprobacion tiene fallos ^(arriba^). Puedes intentar abrir igualmente.
  pause
) else (
  echo ok> "%ROOT%.instalado"
)

:lanzar
echo.
echo Abriendo PixelAnim en el navegador... ^(cierra esta ventana para salir^)
echo La primera generacion descarga ~2,5 GB de modelos a la carpeta "modelos".
"%PY%" "%ROOT%pixelanim.py" web
if errorlevel 1 pause
exit /b 0

:neterr
echo.
echo [ERROR] Fallo una descarga. Revisa la conexion y vuelve a abrir INICIAR.bat ^(continua donde se quedo^).
pause
exit /b 1

:piperr
echo.
echo [ERROR] Fallo instalando librerias. Vuelve a abrir INICIAR.bat; si se repite, envia captura de esta ventana.
pause
exit /b 1
