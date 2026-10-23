"""Motores de detección: nombres, tallas, familia de un peso y tope de los DETR.

Lo que se vigila es lo que rompe una comparación de arquitecturas a mitad de
camino, cuando ya se gastaron horas de GPU:

* pedir un peso que no existe en el catálogo (``yolov11m.pt``, ``rtdetr-m.pt``);
* abrir un RT-DETR con la clase de YOLO, que lo infiere con NMS y da basura;
* que una placa densa sature las 300 consultas de un DETR y el conteo salga
  truncado sin aviso.

No descarga pesos ni usa GPU: los modelos se simulan.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyx.core import motores
from polyx.core.yolo_wrap import YoloModel


# ── Nombres y tallas ────────────────────────────────────────────────
@pytest.mark.parametrize("clave,talla,esperado", [
    ("v8", "m", "yolov8m.pt"),
    ("v11", "m", "yolo11m.pt"),       # sin "v": el catalogo no tiene yolov11m.pt
    ("v8-p2", "s", "yolov8s.pt"),     # la P2 parte de los pesos de v8
    ("rtdetr", "l", "rtdetr-l.pt"),
    ("rtdetr", "x", "rtdetr-x.pt"),
])
def test_peso_base_es_un_nombre_del_catalogo(clave, talla, esperado):
    assert motores.peso_base(clave, talla) == esperado


@pytest.mark.parametrize("talla,esperada", [("n", "l"), ("s", "l"), ("m", "l"),
                                            ("l", "l"), ("x", "x")])
def test_rtdetr_cae_a_la_talla_que_existe(talla, esperada):
    """RT-DETR solo tiene l y x; pedir m no puede hacer fallar la descarga."""
    assert motores.talla_efectiva("rtdetr", talla) == esperada


def test_la_p2_se_describe_con_su_yaml_y_su_origen():
    assert motores.descripcion_base("v8-p2", "m") == "yolov8m-p2.yaml (desde yolov8m.pt)"


def test_familia_desconocida_falla_con_mensaje():
    with pytest.raises(ValueError, match="desconocida"):
        motores.familia("yolo99")


def test_todas_las_familias_estan_en_el_orden_de_la_interfaz():
    assert set(motores.ORDEN) == set(motores.FAMILIAS)


# ── Qué familia trae un modelo cargado ──────────────────────────────
class _Red:
    def __init__(self, yaml_file="", num_queries=None, nombre="DetectionModel"):
        self.yaml = {"yaml_file": yaml_file}
        self.__class__ = type(nombre, (object,), {})
        if num_queries is not None:
            cabeza = type("Cabeza", (), {"num_queries": num_queries})()
            self.model = [cabeza]


class _Modelo:
    def __init__(self, red):
        self.model = red


@pytest.mark.parametrize("yaml_file,esperada", [
    ("yolov8m.yaml", "v8"),
    ("yolo11m.yaml", "v11"),
    ("yolov8m-p2.yaml", "v8-p2"),
    ("", "v8"),   # pesos viejos sin yaml_file: se asume la linea base
])
def test_familia_de_un_yolo(yaml_file, esperada):
    assert motores.familia_de(_Modelo(_Red(yaml_file))) == esperada


def test_familia_y_tope_de_un_rtdetr():
    m = _Modelo(_Red("rtdetr-l.yaml", num_queries=300, nombre="RTDETRDetectionModel"))
    assert motores.familia_de(m) == "rtdetr"
    assert motores.tope_por_pase(m) == 300


def test_rtdetr_publicado_como_detectionmodel_se_reconoce_por_su_cabeza():
    """El rtdetr-l.pt oficial viene como DetectionModel generico: por nombre de
    clase pasaba por YOLO y se habria inferido con NMS."""
    red = _Red("", nombre="DetectionModel")
    red.model = [type("RTDETRDecoder", (), {"num_queries": 300})()]
    m = _Modelo(red)
    assert motores.familia_de(m) == "rtdetr"
    assert motores.tope_por_pase(m) == 300


def test_yolo_no_tiene_tope():
    assert motores.tope_por_pase(_Modelo(_Red("yolov8m.yaml"))) is None


# ── Familias a entrenar ─────────────────────────────────────────────
def test_comparar_respeta_el_orden_fijo_y_no_el_de_marcado():
    from polyx.trainer.state import ModelConfig
    cfg = ModelConfig(comparar_familias=True, familias_comparar=["rtdetr", "v8"])
    assert cfg.familias_a_entrenar() == ["v8", "rtdetr"]


def test_sin_comparar_se_entrena_solo_la_elegida():
    from polyx.trainer.state import ModelConfig
    cfg = ModelConfig(family="v8-p2", comparar_familias=False,
                      familias_comparar=["v8", "v11"])
    assert cfg.familias_a_entrenar() == ["v8-p2"]


# ── Saturación de un DETR ───────────────────────────────────────────
class _Tensor:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Cajas:
    def __init__(self, n, x0=0):
        xs = np.arange(n, dtype=float) * 3 + x0
        self.xyxy = _Tensor(np.stack([xs, xs, xs + 2, xs + 2], axis=1))
        self.conf = _Tensor(np.full(n, 0.9))
        self.cls = _Tensor(np.zeros(n))

    def __len__(self):
        return len(self.conf.numpy())


class _Resultado:
    def __init__(self, n):
        self.boxes = _Cajas(n)


class _DetrFalso:
    """Devuelve siempre `tope` cajas en el pase completo y pocas por tile."""

    def __init__(self, tope, por_tile):
        self.tope, self.por_tile = tope, por_tile
        self.llamadas = []

    def predict(self, source, **kw):
        if isinstance(source, list):
            self.llamadas.append(("tiles", len(source)))
            return [_Resultado(self.por_tile) for _ in source]
        self.llamadas.append(("completa", 1))
        return [_Resultado(self.tope)]


def _modelo_falso(tmp_path, tope, por_tile):
    import cv2
    ruta = tmp_path / "placa.png"
    cv2.imwrite(str(ruta), np.zeros((1800, 1800, 3), np.uint8))
    ym = YoloModel(ruta)
    ym._model = _DetrFalso(tope, por_tile)
    ym.names = {0: "PET"}
    ym.familia, ym.tope = "rtdetr", tope
    return ym, ruta


def test_un_pase_saturado_se_vuelve_a_hacer_troceado(tmp_path):
    """La placa no supera el umbral de troceo, pero el DETR devolvio sus 300."""
    ym, ruta = _modelo_falso(tmp_path, tope=300, por_tile=50)
    reg = {}
    dets = ym.predict_auto(ruta, imgsz=1280, umbral_px=2000, registro=reg)
    tipos = [t for t, _ in ym._model.llamadas]
    assert tipos[0] == "completa" and "tiles" in tipos
    assert reg["troceado"] and reg["por_saturacion"]
    assert not reg.get("saturado"), "los tiles no saturaron: no debe quedar aviso"
    assert len(dets) > 0


def test_un_pase_que_no_satura_no_se_repite(tmp_path):
    ym, ruta = _modelo_falso(tmp_path, tope=300, por_tile=50)
    ym._model.tope = 120   # el pase completo devuelve 120 < 300
    reg = {}
    ym.predict_auto(ruta, imgsz=1280, umbral_px=2000, registro=reg)
    assert [t for t, _ in ym._model.llamadas] == ["completa"]
    assert not reg["troceado"]


def test_si_tambien_saturan_los_tiles_queda_el_aviso(tmp_path):
    ym, ruta = _modelo_falso(tmp_path, tope=300, por_tile=300)
    reg = {}
    ym.predict_auto(ruta, imgsz=1280, umbral_px=2000, registro=reg)
    assert reg["saturado"], "el conteo puede estar truncado y el informe debe saberlo"


def test_yolo_sin_tope_nunca_satura(tmp_path):
    ym, ruta = _modelo_falso(tmp_path, tope=300, por_tile=50)
    ym.tope = None
    reg = {}
    ym.predict_auto(ruta, imgsz=1280, umbral_px=2000, registro=reg)
    assert [t for t, _ in ym._model.llamadas] == ["completa"]
