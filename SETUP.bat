@echo off
setlocal enabledelayedexpansion
title Poly-X - Instalacion

echo.
echo ============================================================
echo   Poly-X  -  Instalacion del entorno
echo ============================================================
echo.

REM ---- 1. Localizar Python 3.11 ----
set "PYEXE="
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
)
if "!PYEXE!"=="" (
    where python >nul 2>&1
    if %errorlevel%==0 (
        for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
    )
)
if "!PYEXE!"=="" (
    echo [ERROR] No se encontro Python 3.11 instalado.
    echo Descargalo desde https://www.python.org/downloads/release/python-3119/
    echo Marca "Add Python to PATH" y "tcl/tk and IDLE" durante la instalacion.
    pause
    exit /b 1
)
echo [OK] Python encontrado en: !PYEXE!

REM ---- 2. Limpiar venv viejo si esta roto ----
if exist .venv (
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo [INFO] El .venv existente esta roto. Recreando...
        rmdir /s /q .venv
    )
)

REM ---- 3. Crear venv si no existe ----
if not exist .venv (
    echo [INFO] Creando entorno virtual .venv ...
    "!PYEXE!" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el venv.
        pause & exit /b 1
    )
)

REM ---- 4. Actualizar pip ----
echo [INFO] Actualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

REM ---- 5. Detectar GPU NVIDIA ----
set "HAS_GPU=0"
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi >nul 2>&1
    if !errorlevel!==0 set "HAS_GPU=1"
)

if "!HAS_GPU!"=="1" (
    echo [OK] GPU NVIDIA detectada. Instalando PyTorch con CUDA 11.8 ^(3-4 GB^)...
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] Sin GPU NVIDIA. Instalando PyTorch CPU...
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM ---- 6. Resto de dependencias ----
echo [INFO] Instalando dependencias del proyecto...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Hubo problemas instalando dependencias.
    pause & exit /b 1
)

echo.
echo ============================================================
echo   Instalacion completada exitosamente
echo   Para iniciar Poly-X: doble clic en iniciar_polyx.bat
echo ============================================================
echo.
pause
