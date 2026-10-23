"""RF-DETR detrás de la interfaz de Ultralytics.

Lo que se vigila:

* que un ``.pth`` se abra con RF-DETR y un ``.pt`` con Ultralytics;
* que la imagen llegue en **RGB**: OpenCV entrega BGR, y con los canales
  cruzados el rojo del Nile Red llegaría azul y el PET dejaría de verse;
* que la salida tenga la forma ``.boxes.xyxy/.conf/.cls`` que lee ``YoloModel``;
* que la resolución pedida se ajuste al múltiplo de 32 que exige el backbone.

No necesita el paquete rfdetr ni GPU: se reemplaza por uno falso.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from polyx.core import motor_rfdetr, motores
from polyx.core.yolo_wrap import YoloModel


class _Detecciones:
    def __init__(self, n):
        self.xyxy = np.array([[10.0 * i, 5.0, 10.0 * i + 4, 9.0] for i in range(n)])
        self.confidence = np.full(n, 0.8)
        self.class_id = np.arange(n) % 3


class _RFFalso:
    vistas = []

    class model_config:
        resolution = 640
        num_queries = 300

    class_names = ["PET", "PP", "LDPE"]

    @classmethod
    def from_checkpoint(cls, ruta, trust_checkpoint=False):
        return cls()

    def predict(self, imagenes, threshold=0.5, include_source_image=True):
        _RFFalso.vistas = [np.array(x) for x in imagenes]
        return [_Detecciones(2) for _ in imagenes]


@pytest.fixture
def rfdetr_falso(monkeypatch):
    mod = types.ModuleType("rfdetr")
    mod.RFDETR = _RFFalso
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    monkeypatch.setattr(motor_rfdetr, "disponible", lambda: True)
    return mod


def test_un_pth_se_abre_con_rfdetr(rfdetr_falso, tmp_path):
    peso = tmp_path / "checkpoint_best_total.pth"
    peso.write_bytes(b"x")
    modelo, familia = motores.cargar(peso)
    assert familia == "rfdetr"
    assert modelo.names == {0: "PET", 1: "PP", 2: "LDPE"}
    assert motores.tope_por_pase(modelo) == 300


def test_la_imagen_llega_en_rgb(rfdetr_falso, tmp_path):
    m = motor_rfdetr.ModeloRFDETR(tmp_path / "x.pth")
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 2] = 200          # rojo en BGR: canal 2
    m.predict([bgr], conf=0.3)
    rgb = _RFFalso.vistas[0]
    assert rgb[0, 0, 0] == 200 and rgb[0, 0, 2] == 0, "el rojo debe quedar en el canal 0"


def test_la_salida_tiene_la_forma_de_ultralytics(rfdetr_falso, tmp_path):
    m = motor_rfdetr.ModeloRFDETR(tmp_path / "x.pth")
    res = m.predict(np.zeros((8, 8, 3), np.uint8), conf=0.3)
    assert len(res) == 1 and len(res[0].boxes) == 2
    assert res[0].boxes.xyxy.cpu().numpy().shape == (2, 4)
    assert list(res[0].boxes.cls.cpu().numpy()) == [0.0, 1.0]


def test_yolomodel_detecta_con_rfdetr_sin_saber_que_motor_es(rfdetr_falso, tmp_path):
    import cv2
    foto = tmp_path / "placa.png"
    cv2.imwrite(str(foto), np.zeros((300, 300, 3), np.uint8))
    peso = tmp_path / "m.pth"
    peso.write_bytes(b"x")
    ym = YoloModel(peso).load()
    dets = ym.predict(foto, conf=0.3)
    assert ym.familia == "rfdetr" and ym.tope == 300
    assert [d.class_name for d in dets] == ["PET", "PP"]


def test_las_cajas_de_fondo_de_rfdetr_se_descartan(rfdetr_falso, tmp_path):
    """El indice extra de fondo (3 con 3 clases) no es ninguna particula."""
    class _ConFondo(_RFFalso):
        def predict(self, imagenes, threshold=0.5, include_source_image=True):
            d = _Detecciones(4)
            d.class_id = np.array([0, 3, 2, 3])
            return [d]
    rfdetr_falso.RFDETR = _ConFondo
    m = motor_rfdetr.ModeloRFDETR(tmp_path / "x.pth")
    res = m.predict(np.zeros((8, 8, 3), np.uint8))
    assert list(res[0].boxes.cls.cpu().numpy()) == [0.0, 2.0]
    assert res[0].boxes.xyxy.cpu().numpy().shape == (2, 4)


def test_sin_instalar_avisa_como_instalarlo(monkeypatch, tmp_path):
    monkeypatch.setattr(motor_rfdetr, "disponible", lambda: False)
    with pytest.raises(RuntimeError, match="instalar_rfdetr"):
        motor_rfdetr.ModeloRFDETR(tmp_path / "x.pth")


@pytest.mark.parametrize("pedida,valida", [(640, 640), (1600, 1600), (1000, 992),
                                           (1010, 1024), (10, 32)])
def test_resolucion_multiplo_de_32(pedida, valida):
    assert motor_rfdetr.resolucion_valida(pedida) == valida


def test_preparar_dataset_no_pisa_un_data_yaml_existente(tmp_path):
    (tmp_path / "dataset.yaml").write_text("train: images/train\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("mio\n", encoding="utf-8")
    motor_rfdetr.preparar_dataset(tmp_path / "dataset.yaml")
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8") == "mio\n"


def test_preparar_dataset_crea_el_data_yaml_que_busca_rfdetr(tmp_path):
    (tmp_path / "dataset.yaml").write_text("train: images/train\n", encoding="utf-8")
    raiz = motor_rfdetr.preparar_dataset(tmp_path / "dataset.yaml")
    assert raiz == tmp_path
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8") == "train: images/train\n"


def test_rfdetr_no_se_entrena_por_ultralytics():
    with pytest.raises(ValueError, match="motor_rfdetr"):
        motores.modelo_para_entrenar("rfdetr", "m")


def test_el_flujo_de_log_entrega_lineas_y_solo_el_ultimo_estado_de_una_barra():
    """Las barras de tqdm reescriben la linea con \r: al log va solo el final."""
    lineas = []
    f = motor_rfdetr._FlujoLog(lineas.append)
    f.write("Epoca 1 ")
    f.write("10%\r50%\r100%\n")
    f.write("┏━━ tabla de rich ━━┓\n")
    f.write("sin salto al final")
    f.flush()
    assert lineas == ["100%", "┏━━ tabla de rich ━━┓", "sin salto al final"]


def test_un_log_que_imprime_no_entra_en_bucle(capsys):
    """Con log=print, print escribe en el mismo stdout redirigido."""
    import contextlib
    f = motor_rfdetr._FlujoLog(print)
    with contextlib.redirect_stdout(f):
        print("linea de rfdetr")
        f.flush()
