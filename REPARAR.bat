@echo off
REM Fuerza a reinstalar/comprobar librerias en el proximo INICIAR.bat
cd /d "%~dp0"
del ".instalado" 2>nul
echo Hecho. Abre INICIAR.bat
pause
