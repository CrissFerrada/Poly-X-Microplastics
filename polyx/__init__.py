"""Poly-X — Suite de detección de microplásticos."""
import os as _os
import sys as _sys

__version__ = "2.0.0"
__author__ = "Cristofher Ferrada"


def _asegurar_salida_estandar() -> None:
    """Garantiza que sys.stdout y sys.stderr existan.

    Cuando Poly-X se abre desde el acceso directo del Escritorio, arranca con
    ``pythonw.exe`` para no dejar una consola negra abierta. pythonw deja
    ``sys.stdout`` y ``sys.stderr`` en ``None``, y el launcher lanza cada modulo
    con ``sys.executable``, asi que los hijos heredan lo mismo.

    Eso mata el entrenamiento: Ultralytics dibuja una barra ``tqdm`` al cachear
    las etiquetas y tqdm escribe en ``sys.stderr``, con lo que salta
    ``AttributeError: 'NoneType' object has no attribute 'write'``.

    El fallo solo aparece la **primera** vez que se usa un dataset, porque en las
    siguientes ya existe el ``.cache`` y esa barra no se dibuja. De ahi que pase
    inadvertido en la maquina donde se desarrolla y reviente en una limpia.

    Se redirige a ``os.devnull``: sin consola no hay donde mostrar ese texto, y
    las metricas del entrenamiento llegan a la interfaz por callbacks, no por la
    salida estandar.
    """
    for nombre in ("stdout", "stderr"):
        if getattr(_sys, nombre, None) is None:
            try:
                destino = open(_os.devnull, "w", encoding="utf-8", errors="replace")
            except OSError:
                continue
            setattr(_sys, nombre, destino)
            # Algunas librerias miran los originales en vez de los actuales.
            if getattr(_sys, f"__{nombre}__", None) is None:
                setattr(_sys, f"__{nombre}__", destino)


_asegurar_salida_estandar()
