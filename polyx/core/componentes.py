"""Componentes opcionales: paquetes que Poly-X no trae de fábrica y se instalan aparte.

Hoy solo hay uno, RF-DETR. Está pensado para que el siguiente (Cellpose, un
exportador, lo que sea) sea **una entrada más en COMPONENTES**: el diálogo de
instalación, la sección de Preferencias y los botones de cada módulo leen de
aquí y no saben qué están instalando.

Cada componente declara tres cosas:

* el archivo de requisitos, que fija versiones contra ``requirements.txt``. Un
  ``pip install paquete`` suelto puede arrastrar dependencias que rompen lo que
  ya funciona (RF-DETR subía numpy a 2.x y Ultralytics dejaba de entrenar);
* el módulo cuya presencia dice si está instalado;
* una verificación que se ejecuta **después** de instalar, en un proceso nuevo,
  y que comprueba que no se movió nada de lo que necesita el resto del programa.

Sin Qt: lo usan tanto la interfaz como los tests.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Componente:
    clave: str
    nombre: str
    descripcion: str        # en español; la interfaz la pasa por tr()
    requisitos: str         # archivo relativo a la raíz del programa
    modulo: str             # import que dice si está instalado
    verificacion: str       # código Python que debe terminar sin error
    tamano: str = ""        # orientación para el usuario, no se mide


# numpy 1.x es la condición que importa: con numpy 2 Ultralytics 8.3.40 muere
# al validar la primera época (np.trapz), que es el peor momento para saberlo.
_VERIFICA_NUMPY = (
    "import numpy, ultralytics; "
    "assert numpy.__version__.startswith('1.'), "
    "'numpy ' + numpy.__version__ + ': Ultralytics necesita numpy 1.x'")

COMPONENTES: Dict[str, Componente] = {
    "rfdetr": Componente(
        clave="rfdetr",
        nombre="RF-DETR",
        descripcion="Motor de detección de Roboflow (transformer con backbone "
                    "DINOv2). Hace falta para entrenar o usar modelos RF-DETR (.pth).",
        requisitos="requirements-rfdetr.txt",
        modulo="rfdetr",
        verificacion=_VERIFICA_NUMPY + "; import rfdetr, pytorch_lightning, pyarrow; "
                     "print('rfdetr OK')",
        tamano="unos 50 paquetes, varios cientos de MB",
    ),
}


def raiz() -> Path:
    """Carpeta del programa (la que tiene requirements.txt)."""
    return Path(__file__).resolve().parents[2]


def componente(clave: str) -> Componente:
    try:
        return COMPONENTES[clave]
    except KeyError:
        raise ValueError(f"componente opcional desconocido: {clave!r}") from None


def instalado(clave: str) -> bool:
    """¿Está el componente en el entorno de Poly-X?

    Se invalidan las cachés de importación porque el instalador corre en otro
    proceso: sin eso, un módulo recién instalado no se ve hasta reiniciar.
    """
    importlib.invalidate_caches()
    return importlib.util.find_spec(componente(clave).modulo) is not None


def interprete() -> Path:
    """El Python del entorno, con consola.

    Poly-X corre con ``pythonw.exe`` para no abrir una ventana negra. Para pip
    se usa el ``python.exe`` de al lado: es el mismo entorno, y su salida se
    puede leer y mostrar en el diálogo.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        hermano = exe.with_name("python.exe")
        if hermano.exists():
            return hermano
    return exe


def comando_instalar(clave: str) -> List[str]:
    c = componente(clave)
    return [str(interprete()), "-m", "pip", "install", "--disable-pip-version-check",
            "-r", str(raiz() / c.requisitos)]


def comando_verificar(clave: str) -> List[str]:
    return [str(interprete()), "-c", componente(clave).verificacion]
