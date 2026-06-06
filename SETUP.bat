@echo off
setlocal enabledelayedexpansion
title Poly-X - Instalacion
REM Trabajar siempre desde la carpeta donde esta este SETUP.bat
cd /d "%~dp0"
set "SRC=%cd%"

echo.
echo ============================================================
echo   Poly-X  -  Instalacion del entorno
echo ============================================================
echo.
echo Carpeta de origen (lo que descargaste de GitHub):
echo   %SRC%
echo.

REM ============================================================
REM   0. Donde instalar
REM ============================================================
echo Donde quieres INSTALAR Poly-X?
echo   - Pulsa ENTER para instalarlo AQUI MISMO (recomendado)
echo   - O escribe/pega la ruta de otra carpeta y pulsa ENTER
echo.
set "DEST="
set /p "DEST=Carpeta de instalacion: "
REM Quitar comillas si las pego el usuario
if defined DEST set "DEST=!DEST:"=!"
if not defined DEST set "DEST=%SRC%"

if /i not "!DEST!"=="%SRC%" (
    echo.
    echo [INFO] Se instalara en: !DEST!
    if not exist "!DEST!" mkdir "!DEST!"
    echo [INFO] Copiando archivos del programa ^(esto puede tardar un momento^)...
    robocopy "%SRC%" "!DEST!" /E /XD .venv .git __pycache__ runs /XF *.log /NFL /NDL /NJH /NJS /NP >nul
    if errorlevel 8 (
        echo [ERROR] No se pudieron copiar los archivos a "!DEST!".
        pause & exit /b 1
    )
    cd /d "!DEST!"
)
set "INSTALL=%cd%"
echo.
echo [OK] Carpeta de instalacion: !INSTALL!
echo.

REM ============================================================
REM   1. Localizar Python 3.11
REM ============================================================
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

REM ============================================================
REM   2. Limpiar venv viejo si esta roto
REM ============================================================
if exist .venv (
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo [INFO] El .venv existente esta roto. Recreando...
        rmdir /s /q .venv
    )
)

REM ============================================================
REM   3. Crear venv si no existe
REM ============================================================
if not exist .venv (
    echo [INFO] Creando entorno virtual .venv ...
    "!PYEXE!" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el venv.
        pause & exit /b 1
    )
)

REM ============================================================
REM   4. Actualizar pip
REM ============================================================
echo [INFO] Actualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

REM ============================================================
REM   5. Detectar GPU NVIDIA
REM ============================================================
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

REM ============================================================
REM   6. Resto de dependencias
REM ============================================================
echo [INFO] Instalando dependencias del proyecto...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Hubo problemas instalando dependencias.
    pause & exit /b 1
)

REM ============================================================
REM   7. Comprobacion rapida de que todo importa
REM ============================================================
echo [INFO] Verificando la instalacion...
.venv\Scripts\python.exe -c "import PySide6, ultralytics, cv2, numpy, matplotlib; import PySide6.QtWebEngineWidgets" 2>nul
if errorlevel 1 (
    echo [AVISO] Algunas dependencias no se importaron correctamente.
    echo         El programa podria fallar. Revisa los mensajes de arriba.
) else (
    echo [OK] Todas las dependencias se importan correctamente ^(incluida la exportacion a PDF^).
)

echo.
echo ============================================================
echo   Instalacion completada
echo ============================================================
echo.
echo  Poly-X quedo instalado en:
echo     !INSTALL!
echo.
echo  COMO INICIARLO:
echo     1) Abre la carpeta:  !INSTALL!
echo     2) Doble clic en:    iniciar_polyx.bat
echo.
echo  NOTA: los modelos entrenados (.pt) NO vienen en la descarga de
echo        GitHub por su tamano. Copia tu archivo .pt dentro de la
echo        carpeta  models\  para usar el Detector y el Visor.
echo.

REM ---- Acceso directo en el Escritorio ----
REM (Se usa 'goto' para que los parentesis del comando PowerShell no
REM  interfieran con el parser de bloques de cmd.)
set "MKLINK="
set /p "MKLINK=Crear un acceso directo 'Poly-X' en el Escritorio? (S/N): "
if /i not "!MKLINK!"=="S" goto :skip_shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut((Join-Path $d 'Poly-X.lnk')); $s.TargetPath='!INSTALL!\iniciar_polyx.bat'; $s.WorkingDirectory='!INSTALL!'; $s.Save()"
if errorlevel 1 (echo [AVISO] No se pudo crear el acceso directo.) else (echo [OK] Acceso directo 'Poly-X' creado en el Escritorio.)
:skip_shortcut

echo.
set "RUNNOW="
set /p "RUNNOW=Quieres iniciar Poly-X ahora? (S/N): "
if /i not "!RUNNOW!"=="S" goto :no_run
start "" "!INSTALL!\iniciar_polyx.bat"
exit /b 0
:no_run

echo.
echo Listo. Cuando quieras, abre  iniciar_polyx.bat  en:
echo   !INSTALL!
echo.
pause
