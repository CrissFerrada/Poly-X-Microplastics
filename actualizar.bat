@echo off
title Poly-X - Actualizar
setlocal
cd /d "%~dp0"

REM Ruta de esta carpeta sin la barra final (para pasarla a PowerShell)
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se encontro PowerShell; no se puede actualizar automaticamente.
    echo Descarga la ultima version manualmente desde:
    echo   https://github.com/CrissFerrada/Poly-X-Microplastics
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\actualizar.ps1" -InstallDir "%HERE%"
