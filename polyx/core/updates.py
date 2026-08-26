"""Aviso de version nueva disponible en GitHub.

La comprobacion corre en un hilo aparte y nunca bloquea la interfaz: si no hay
internet, si GitHub responde lento o si la instalacion no vino de un ZIP, el
launcher simplemente no muestra nada. Un aviso de actualizacion no justifica
demorar el arranque ni molestar con un error.

Quien aplica la actualizacion es un script externo, distinto en cada sistema
(`actualizar.bat` en Windows, `actualizar_macOS.command` en Mac); aca solo se
detecta y se avisa. La marca de version instalada es el SHA del ultimo commit,
guardado en `.polyx_version` por el instalador y por el propio actualizador.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .paths import ROOT
from .plataforma import nombre_actualizador

REPO = "CrissFerrada/Poly-X-Microplastics"
RAMA = "main"
API = f"https://api.github.com/repos/{REPO}/commits/{RAMA}"
ARCHIVO_VERSION = ROOT / ".polyx_version"
# El nombre depende del sistema: cada uno necesita su propio shell para
# descomprimir y sobrescribir. Ver plataforma.nombre_actualizador().
ACTUALIZADOR = ROOT / nombre_actualizador()

# Corto a proposito: si GitHub no responde rapido, no vale la pena insistir.
TIMEOUT_S = 6


def version_instalada() -> str:
    """Return the commit SHA recorded at install time, or '' if unknown."""
    try:
        return ARCHIVO_VERSION.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def version_remota() -> str:
    """Return the latest commit SHA on the tracked branch, or '' on failure."""
    req = urllib.request.Request(API, headers={"User-Agent": "PolyX-Updater"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return (json.load(r).get("sha") or "").strip()
    except Exception:
        # Sin internet, GitHub caido, proxy, rate limit... nada de esto merece
        # interrumpir al usuario: se calla y no muestra aviso.
        return ""


class BuscadorActualizaciones(QObject):
    """Comprueba en segundo plano si hay una version nueva en GitHub."""

    hay_version_nueva = Signal(str)  # SHA remoto, abreviado

    def buscar(self) -> None:
        """Lanza la comprobacion sin bloquear; emite la senal solo si procede."""
        hilo = threading.Thread(target=self._trabajo, daemon=True)
        hilo.start()

    def _trabajo(self) -> None:
        local = version_instalada()
        # Sin marca local no se puede comparar. Pasa en instalaciones de
        # desarrollo (clon de git): ahi el aviso sobra, se actualiza con git.
        if not local:
            return
        remota = version_remota()
        if remota and remota != local:
            self.hay_version_nueva.emit(remota[:7])


def puede_actualizar() -> bool:
    """True si existe el actualizador que aplica los cambios."""
    return ACTUALIZADOR.exists()
