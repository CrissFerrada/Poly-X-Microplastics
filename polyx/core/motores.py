"""Motores de detección: qué arquitectura hay detrás de un archivo de pesos.

Poly-X nació con YOLO v8/v11 y todo el programa infiere a través de
``YoloModel``. Este módulo es lo único que sabe que hay otras arquitecturas:
dado un archivo de pesos, carga la clase correcta, y dado un nombre de familia,
arma el modelo base para entrenar. Así el Detector, el Visor, el Etiquetador y
el Entrenador aceptan un motor nuevo sin cambiar su propio código.

Familias disponibles:

=========  ============================================================
``v8``     YOLOv8. La línea base del proyecto.
``v11``    YOLO11.
``v8-p2``  YOLOv8 con cabeza P2 (stride 4): una escala más fina, pensada
           para objetos pequeños. Las partículas miden 17-18 px a imgsz
           1600, donde la cabeza P3 de YOLO ya trabaja a 8 px por celda.
``rtdetr`` RT-DETR (transformer, sin NMS). Solo existe en tallas l y x.
``rfdetr`` RF-DETR de Roboflow (transformer con backbone DINOv2). Opcional:
           se instala aparte (``core.motor_rfdetr``) y guarda ``.pth``.
=========  ============================================================

Todas comparten el formato de dataset YOLO (``.txt`` con ``clase cx cy w h``),
así que se entrenan y se comparan sobre los mismos datos sin convertir nada.

**Tope por pase de los DETR.** RT-DETR no usa NMS: propone un número fijo de
consultas (300 de fábrica) y se queda con las que superan el umbral. Una placa
de desembocadura pasa de 500 partículas, así que en un pase único el modelo
*no puede* devolverlas todas. ``tope_por_pase()`` lo expone para que la
inferencia trocee la imagen cuando el pase sale saturado, en vez de truncar en
silencio el conteo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Familia:
    clave: str
    nombre: str
    tallas: str          # tallas que existen, en orden
    base: str            # nombre del peso base publicado; {t} = talla
    yaml: str = ""       # arquitectura distinta a la del peso base (P2)
    descripcion: str = ""


FAMILIAS: Dict[str, Familia] = {
    "v8": Familia("v8", "YOLOv8", "nsmlx", "yolov8{t}.pt",
                  descripcion="La línea base del proyecto."),
    "v11": Familia("v11", "YOLO11", "nsmlx", "yolo11{t}.pt",
                   descripcion="Sucesora de v8, con menos parámetros por talla."),
    "v8-p2": Familia("v8-p2", "YOLOv8-P2", "nsmlx", "yolov8{t}.pt",
                     yaml="yolov8{t}-p2.yaml",
                     descripcion="Cabeza extra de alta resolución para partículas "
                                 "diminutas. Más lenta y con más memoria."),
    "rtdetr": Familia("rtdetr", "RT-DETR", "lx", "rtdetr-{t}.pt",
                      descripcion="Transformer sin NMS. Solo tallas l y x; "
                                  "se trocea solo cuando una placa satura sus "
                                  "300 detecciones por pase."),
    "rfdetr": Familia("rfdetr", "RF-DETR", "nsml", "rf-detr-{t}",
                      descripcion="Transformer de Roboflow con backbone DINOv2. "
                                  "Opcional: se instala aparte. Sin curvas en vivo "
                                  "ni elección de best_real.pt."),
}

# Orden en que se ofrecen en la interfaz.
ORDEN = ("v8", "v11", "v8-p2", "rtdetr", "rfdetr")

# Familias que no entrenan con Ultralytics y siguen su propio camino.
FUERA_DE_ULTRALYTICS = ("rfdetr",)


def familia(clave: str) -> Familia:
    try:
        return FAMILIAS[clave]
    except KeyError:
        raise ValueError(f"familia de motor desconocida: {clave!r}") from None


def talla_efectiva(clave: str, talla: str) -> str:
    """La talla que realmente existe para esa familia.

    RT-DETR solo se publica en l y x: pedir "m" no puede fallar a mitad de un
    entrenamiento de comparación, así que cae a la más cercana por arriba.
    """
    tallas = familia(clave).tallas
    if talla in tallas:
        return talla
    orden = "nsmlx"
    pos = orden.index(talla) if talla in orden else 2
    for t in orden[pos:]:
        if t in tallas:
            return t
    return tallas[-1]


def peso_base(clave: str, talla: str) -> str:
    """Nombre del peso base, tal como lo descarga Ultralytics."""
    return familia(clave).base.format(t=talla_efectiva(clave, talla))


def descripcion_base(clave: str, talla: str) -> str:
    """Lo que se va a entrenar, en una frase para el log y la interfaz."""
    f = familia(clave)
    t = talla_efectiva(clave, talla)
    if f.yaml:
        return f"{f.yaml.format(t=t)} (desde {f.base.format(t=t)})"
    return f.base.format(t=t)


def modelo_para_entrenar(clave: str, talla: str):
    """Modelo de Ultralytics listo para ``.train()``, con pesos preentrenados.

    La P2 no tiene peso publicado: se construye su arquitectura desde el YAML y
    se le transfieren los pesos de la v8 de la misma talla. Todo lo que coincide
    (backbone y cuello) llega preentrenado; solo la cabeza nueva parte de cero.
    """
    f = familia(clave)
    t = talla_efectiva(clave, talla)
    if clave in FUERA_DE_ULTRALYTICS:
        raise ValueError(f"{f.nombre} no se entrena con Ultralytics: "
                         "usa core.motor_rfdetr.entrenar()")
    if clave == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(f.base.format(t=t))
    from ultralytics import YOLO
    if f.yaml:
        return YOLO(f.yaml.format(t=t)).load(f.base.format(t=t))
    return YOLO(f.base.format(t=t))


def _es_rtdetr(modelo_torch) -> bool:
    """¿La red es un RT-DETR?

    No basta el nombre de la clase: el ``rtdetr-l.pt`` publicado por
    Ultralytics viene serializado como ``DetectionModel`` genérico, igual que
    un YOLO. Lo inequívoco es la cabeza, que en un DETR es ``RTDETRDecoder``.
    """
    if modelo_torch is None:
        return False
    if type(modelo_torch).__name__.startswith("RTDETR"):
        return True
    try:
        return type(modelo_torch.model[-1]).__name__ == "RTDETRDecoder"
    except Exception:
        return False


def familia_de(modelo) -> str:
    """Familia de un modelo de Ultralytics ya cargado."""
    red = getattr(modelo, "model", None)
    if _es_rtdetr(red):
        return "rtdetr"
    yaml = getattr(red, "yaml", None) or {}
    archivo = str(yaml.get("yaml_file", "")).lower() if isinstance(yaml, dict) else ""
    if "p2" in archivo:
        return "v8-p2"
    if "11" in archivo:
        return "v11"
    return "v8"


def cargar_ultralytics(ruta: str | Path) -> Tuple[object, str]:
    """Carga un ``.pt`` de Ultralytics con la clase que le corresponde.

    ``YOLO(ruta)`` abre también un checkpoint de RT-DETR, pero lo infiere con el
    predictor de YOLO, que aplica NMS sobre salidas que no lo esperan y da
    cajas sin sentido. Por eso se mira qué red trae y, si es un DETR, se vuelve
    a abrir con ``RTDETR``.
    """
    from ultralytics import YOLO
    m = YOLO(str(ruta))
    if _es_rtdetr(getattr(m, "model", None)):
        from ultralytics import RTDETR
        m = RTDETR(str(ruta))
    return m, familia_de(m)


def cargar(ruta: str | Path) -> Tuple[object, str]:
    """Carga cualquier peso que Poly-X sepa usar: ``.pt`` de Ultralytics o
    ``.pth`` de RF-DETR. Devuelve (modelo con ``.predict``, familia)."""
    from . import motor_rfdetr
    if motor_rfdetr.es_peso_rfdetr(ruta):
        return motor_rfdetr.ModeloRFDETR(ruta), "rfdetr"
    return cargar_ultralytics(ruta)


# Extensiones que Poly-X sabe abrir como peso de un modelo.
EXTENSIONES_PESO = (".pt", ".pth")


def es_peso(ruta: str | Path) -> bool:
    return Path(ruta).suffix.lower() in EXTENSIONES_PESO


# Filtro de los dialogos de abrir pesos, igual en los cuatro modulos.
FILTRO_PESOS = "Pesos de modelo (*.pt *.pth);;YOLO / RT-DETR (*.pt);;RF-DETR (*.pth)"


def tope_por_pase(modelo) -> Optional[int]:
    """Máximo de detecciones que el modelo puede devolver en un pase, o None.

    Solo los DETR tienen tope: es su número de consultas. YOLO no lo tiene más
    allá de ``max_det``, que Poly-X ya sube a 5000.
    """
    if getattr(modelo, "familia", "") == "rfdetr":
        return getattr(modelo, "tope", 300)
    red = getattr(modelo, "model", None)
    if not _es_rtdetr(red):
        return None
    try:
        return int(red.model[-1].num_queries)
    except Exception:
        return 300


def nombre_de(clave: str) -> str:
    return FAMILIAS[clave].nombre if clave in FAMILIAS else clave
