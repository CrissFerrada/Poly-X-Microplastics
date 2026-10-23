@echo off
title Poly-X - Instalar el motor RF-DETR
REM Motor opcional. Poly-X funciona sin el; solo hace falta para entrenar o
REM usar modelos RF-DETR (.pth).
cd /d "%~dp0"

echo.
echo ============================================================
echo   Poly-X  -  Motor opcional RF-DETR
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No encuentro .venv. Ejecuta primero SETUP.bat.
    pause
    exit /b 1
)

echo Se instalan unos 50 paquetes ^(varios cientos de MB^).
echo numpy, OpenCV y Ultralytics quedan en las versiones de Poly-X.
echo.
".venv\Scripts\python.exe" -m pip install -r requirements-rfdetr.txt
if errorlevel 1 (
    echo.
    echo [ERROR] La instalacion fallo. Revisa el mensaje de arriba.
    pause
    exit /b 1
)

REM Comprobar que no se movio nada de lo que necesita YOLO: con numpy 2 el
REM entrenamiento de Ultralytics 8.3.40 muere al validar la primera epoca.
".venv\Scripts\python.exe" -c "import numpy, ultralytics, rfdetr; assert numpy.__version__.startswith('1.'), 'numpy ' + numpy.__version__; print('[OK] RF-DETR instalado - numpy', numpy.__version__, '- ultralytics', ultralytics.__version__)"
if errorlevel 1 (
    echo.
    echo [ERROR] La instalacion cambio numpy a 2.x. Avisa antes de entrenar.
    pause
    exit /b 1
)
echo.
echo Listo. Cierra y vuelve a abrir Poly-X para que aparezca RF-DETR.
pause
