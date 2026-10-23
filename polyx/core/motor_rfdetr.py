"""RF-DETR (Roboflow) como motor de Poly-X. Opcional: no viene en requirements.txt.

RF-DETR no es de Ultralytics, así que no entra por ``YOLO()``/``RTDETR()``.
Este módulo lo envuelve para que **por fuera se comporte igual**: un objeto con
``.predict(source, conf=..., ...)`` que devuelve resultados con ``.boxes.xyxy``,
``.boxes.conf`` y ``.boxes.cls``. Con eso, ``YoloModel`` (troceado, fallback,
saturación, conteo) funciona sin saber qué motor tiene debajo.

**Instalación aparte, y con versiones fijadas.** ``rfdetr[train]`` sube por
su cuenta numpy a 2.x y OpenCV a 5.x, y Ultralytics 8.3.40 se rompe con numpy 2
(usa ``np.trapz`` al validar). ``requirements-rfdetr.txt`` lo instala
restringido a las versiones del proyecto; nunca con un ``pip install rfdetr``
suelto.

Diferencias que el usuario debe conocer, y que el informe declara:

* **Resolución fija por modelo.** RF-DETR infiere a la resolución con que se
  entrenó (múltiplo de 32); el ``imgsz`` del Detector no la cambia. El troceado
  sigue funcionando: cada tile se reescala a esa resolución.
* **Tope de 300 detecciones por pase**, como RT-DETR. ``YoloModel`` trocea solo
  si un pase satura.
* **Espera RGB.** OpenCV entrega BGR; aquí se convierte, porque con los canales
  cruzados el rojo del Nile Red se volvería azul y el modelo no vería el PET.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

EXTENSIONES = (".pth",)

# Talla de Poly-X -> clase de RF-DETR. No hay "x": cae a Large.
_CLASES = {"n": "RFDETRNano", "s": "RFDETRSmall", "m": "RFDETRMedium",
           "l": "RFDETRLarge", "x": "RFDETRLarge"}
_PASO_RESOLUCION = 32   # patch_size (16) x num_windows (2) en todas las tallas


def disponible() -> bool:
    """¿Está instalado el paquete ``rfdetr``?"""
    # Se instala desde la propia interfaz, en otro proceso: sin invalidar la
    # cache de importacion no se veria hasta reiniciar.
    importlib.invalidate_caches()
    return importlib.util.find_spec("rfdetr") is not None


def es_peso_rfdetr(ruta: str | Path) -> bool:
    return Path(ruta).suffix.lower() in EXTENSIONES


MENSAJE_NO_INSTALADO = (
    "RF-DETR no está instalado. Ejecuta instalar_rfdetr.bat (Windows) o "
    "instalar_rfdetr_macOS.command (Mac) y vuelve a abrir Poly-X.")


def resolucion_valida(imgsz: int) -> int:
    """El múltiplo de 32 más cercano a ``imgsz``, como exige el backbone."""
    return max(_PASO_RESOLUCION,
               int(round(int(imgsz) / _PASO_RESOLUCION)) * _PASO_RESOLUCION)


# ────────────────────────────────────────────────────────────────────
# Resultados con la forma de los de Ultralytics
# ────────────────────────────────────────────────────────────────────
class _Arreglo:
    """Imita un tensor lo justo: ``.cpu().numpy()``."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Cajas:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arreglo(np.asarray(xyxy, dtype=float).reshape(-1, 4))
        self.conf = _Arreglo(np.asarray(conf, dtype=float).reshape(-1))
        self.cls = _Arreglo(np.asarray(cls, dtype=float).reshape(-1))

    def __len__(self):
        return len(self.conf.numpy())


class _Resultado:
    def __init__(self, detecciones, n_clases: int):
        cls = np.asarray(detecciones.class_id).reshape(-1)
        # RF-DETR reserva un indice extra para "fondo" (``_bg_sentinel`` en su
        # predict): con un modelo poco entrenado aparecen cajas con ese id, que
        # no son ninguna particula. Solo se conservan las clases del dataset.
        ok = (cls >= 0) & (cls < n_clases)
        self.boxes = _Cajas(np.asarray(detecciones.xyxy).reshape(-1, 4)[ok],
                            np.asarray(detecciones.confidence).reshape(-1)[ok],
                            cls[ok])


def _a_rgb(img) -> np.ndarray:
    if isinstance(img, (str, Path)):
        import cv2
        bgr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"no se pudo leer la imagen: {img}")
        img = bgr
    # Todo lo que llega como arreglo viene de cv2 (los tiles de predict_sliced).
    return np.ascontiguousarray(np.asarray(img)[..., ::-1])


class ModeloRFDETR:
    """Un checkpoint ``.pth`` de RF-DETR con la interfaz de un modelo de Ultralytics."""

    familia = "rfdetr"

    def __init__(self, ruta: str | Path):
        if not disponible():
            raise RuntimeError(MENSAJE_NO_INSTALADO)
        from rfdetr import RFDETR
        # trust_checkpoint: el .pth lo produjo este mismo programa al entrenar.
        self._m = RFDETR.from_checkpoint(str(ruta), trust_checkpoint=True)
        self.names: Dict[int, str] = dict(enumerate(self._m.class_names))
        self.resolucion = int(getattr(self._m.model_config, "resolution", 0) or 0)
        self.tope = int(getattr(self._m.model_config, "num_queries", 300) or 300)

    def predict(self, source, conf: float = 0.25, **_ignorados) -> List[_Resultado]:
        """Misma llamada que ``YOLO.predict``. ``imgsz``, ``iou`` y ``device`` se
        ignoran: la resolución es la del modelo, no hay NMS y el dispositivo lo
        elige RF-DETR."""
        lote = source if isinstance(source, list) else [source]
        imagenes = [_a_rgb(x) for x in lote]
        salida = self._m.predict(imagenes, threshold=float(conf),
                                 include_source_image=False)
        if not isinstance(salida, list):
            salida = [salida]
        return [_Resultado(d, len(self.names)) for d in salida]


# ────────────────────────────────────────────────────────────────────
# Entrenamiento
# ────────────────────────────────────────────────────────────────────
class _FlujoLog(io.TextIOBase):
    """Archivo de texto que entrega cada línea completa a una función de log.

    Las barras de progreso reescriben la misma línea con ``\\r`` decenas de
    veces por segundo; de cada línea solo se envía el último estado, para no
    inundar el log del Entrenador.
    """

    encoding = "utf-8"

    def __init__(self, log: Callable[[str], None]):
        super().__init__()
        self._log = log
        self._buf = ""
        self._dentro = False

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def _entregar(self, linea: str) -> None:
        # Si la funcion de log escribe a su vez en stdout (un print, por
        # ejemplo), volveria aqui y se llamaria a si misma sin fin.
        self._dentro = True
        try:
            self._log(linea)
        finally:
            self._dentro = False

    def write(self, s: str) -> int:
        if self._dentro:
            consola = sys.__stdout__
            if consola is not None:
                try:
                    consola.write(s)
                except Exception:
                    pass
            return len(s)
        self._buf += s
        while "\n" in self._buf:
            linea, self._buf = self._buf.split("\n", 1)
            linea = linea.rsplit("\r", 1)[-1].rstrip()
            if linea:
                self._entregar(linea)
        return len(s)

    def flush(self) -> None:
        if self._dentro:
            return
        resto = self._buf.rsplit("\r", 1)[-1].rstrip()
        self._buf = ""
        if resto:
            self._entregar(resto)


def preparar_dataset(yaml_dataset: str | Path) -> Path:
    """Carpeta del dataset lista para RF-DETR, que busca ``data.yaml`` en la raíz.

    Los datasets de Poly-X se llaman ``dataset.yaml``. Se deja al lado una copia
    llamada ``data.yaml`` (solo si no existe, para no pisar una del usuario);
    las rutas son relativas, así que la copia apunta a las mismas imágenes.
    """
    yaml_dataset = Path(yaml_dataset)
    raiz = yaml_dataset.parent
    destino = raiz / "data.yaml"
    if yaml_dataset.name not in ("data.yaml", "data.yml") and not destino.exists():
        shutil.copyfile(yaml_dataset, destino)
    return raiz


def entrenar(yaml_dataset: str | Path, talla: str, salida: str | Path, *,
             epochs: int, batch: int, imgsz: int, lr: Optional[float] = None,
             log: Callable[[str], None] = print) -> Path:
    """Entrena RF-DETR sobre un dataset YOLO de Poly-X. Devuelve el mejor ``.pth``."""
    if not disponible():
        raise RuntimeError(MENSAJE_NO_INSTALADO)
    import rfdetr
    clase = getattr(rfdetr, _CLASES.get(talla, "RFDETRMedium"))
    raiz = preparar_dataset(yaml_dataset)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    res = resolucion_valida(imgsz)
    if res != int(imgsz):
        log(f"[INFO] RF-DETR necesita una resolución múltiplo de {_PASO_RESOLUCION}: "
            f"se usa {res} en vez de {imgsz}.")
    log(f"[INFO] RF-DETR {clase.__name__} · resolución {res} · batch {batch} · "
        f"{epochs} épocas")
    modelo = clase()
    kwargs = dict(dataset_dir=str(raiz), dataset_file="yolo", epochs=int(epochs),
                  batch_size=int(batch), output_dir=str(salida), resolution=res,
                  # Escala fija. El multiescala de fabrica entrenaba a 800 px
                  # con 640 pedidos, y la premisa del dataset es que la
                  # particula mida lo mismo (17-18 px a imgsz 1600) en todas
                  # las fuentes. Ademas, comparar motores exige la misma escala.
                  multi_scale=False, expanded_scales=False,
                  # Windows y macOS crean los procesos del DataLoader por spawn:
                  # desde el hilo del Entrenador eso relanza el modulo principal
                  # y falla. Un solo proceso es mas lento pero siempre funciona.
                  num_workers=0 if sys.platform in ("win32", "darwin") else 2)
    if lr:
        kwargs["lr"] = float(lr)
    # RF-DETR escribe tablas con rich y barras con tqdm. En la consola de
    # Windows (cp1252) esos caracteres revientan el entrenamiento, y bajo
    # pythonw no hay consola en absoluto. Se manda todo al log de Poly-X.
    flujo = _FlujoLog(log)
    with contextlib.redirect_stdout(flujo), contextlib.redirect_stderr(flujo):
        try:
            modelo.train(**kwargs)
        finally:
            flujo.flush()
    for nombre in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                   "checkpoint_best_regular.pth", "checkpoint.pth"):
        if (salida / nombre).exists():
            return salida / nombre
    raise FileNotFoundError(f"RF-DETR terminó sin dejar checkpoint en {salida}")
