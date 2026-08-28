"""Que el informe salga de verdad en los dos idiomas.

Hay un modo de fallo que no se nota nunca al programar: si una cadena se
envuelve como ``{tr('...')}`` dentro de una cadena que NO es f-string, Python
no protesta y el documento sale con el codigo impreso tal cual:

    <h3>2.1 {tr('Equipo de computo')}</h3>

Se ve solo abriendo el HTML, y paso tres veces al traducir el informe. Aqui se
comprueba sobre el documento generado, que es donde el fallo existe.

Tambien se vigila lo contrario: que pedir ingles no devuelva un informe en
español porque la clave del diccionario dejo de coincidir con la del codigo
-- basta un espacio o un salto de linea de diferencia para que tr() caiga en
silencio al original.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from polyx.core.yolo_wrap import Detection


def _estado_minimo(tmp_path):
    """Un DetectorState con una deteccion medida sobre una foto sintetica."""
    import cv2
    from polyx.detector.state import DetectorState, ImageResult, ModelSlot
    from polyx.core import morfologia

    img = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(img, (160, 130), 22, (60, 210, 240), -1)
    ruta = tmp_path / "placa.png"
    cv2.imwrite(str(ruta), img)

    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=138, y1=108, x2=182, y2=152)
    morfologia.aplicar_a_deteccion(d, img, 30.0)
    d.numero = 1

    st = DetectorState()
    st.model_slots[0] = ModelSlot(alias="m1", path=Path("fake.pt"))
    st.results = {0: [ImageResult(image_path=ruta, model_idx=0,
                                  predictions=[d], gt=[], has_gt=False)]}
    return st


def _informe(tmp_path, lang):
    """Genera el informe con el idioma pedido y devuelve su HTML."""
    os.environ["POLYX_IDIOMA"] = lang
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.core.report_html as rh
    importlib.reload(rh)
    salida = tmp_path / f"informe_{lang}.html"
    rh.generate_report(_estado_minimo(tmp_path), salida,
                       secciones=list(rh.IDS_SECCIONES))
    return salida.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _restaurar_idioma():
    previo = os.environ.get("POLYX_IDIOMA")
    yield
    if previo is None:
        os.environ.pop("POLYX_IDIOMA", None)
    else:
        os.environ["POLYX_IDIOMA"] = previo
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.core.report_html as rh
    importlib.reload(rh)


@pytest.mark.parametrize("lang", ["es", "en"])
def test_no_queda_codigo_sin_evaluar_en_el_documento(tmp_path, lang):
    """El fallo de envolver en tr() una cadena que no era f-string."""
    html = _informe(tmp_path, lang)
    assert "{tr(" not in html, (
        "el documento lleva un {tr(...)} impreso: esa cadena se envolvio "
        "dentro de una cadena normal, no de una f-string")


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_idioma_del_documento_es_el_pedido(tmp_path, lang):
    html = _informe(tmp_path, lang)
    assert f"<html lang='{lang}'" in html


def test_en_ingles_no_salen_los_rotulos_en_español(tmp_path):
    """Si una clave del diccionario deja de coincidir con la del codigo, tr()
    devuelve el español sin avisar y el informe queda mezclado."""
    html = _informe(tmp_path, "en")
    for rotulo_es, rotulo_en in [("<th>Parámetro</th>", "Parameter"),
                                 ("Morfotipo", "Morphotype"),
                                 ("<td>mínimo</td>", "minimum")]:
        assert rotulo_es not in html, f"quedo en español: {rotulo_es}"
        assert rotulo_en in html, f"falta la traduccion: {rotulo_en}"


def test_en_español_siguen_los_rotulos_en_español(tmp_path):
    """La otra direccion: traducir no debe romper el idioma original."""
    html = _informe(tmp_path, "es")
    assert "<th>Parámetro</th>" in html
    assert "Parameter" not in html


def _texto_visible(html: str) -> list[str]:
    """Lineas de texto del documento, sin marcado ni imagenes embebidas."""
    import re
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    h = re.sub(r"data:image[^\"']+", "", h)
    h = re.sub(r"<[^>]+>", "\n", h)
    for a, b in [("&nbsp;", " "), ("&middot;", "·"), ("&aacute;", "á"),
                 ("&eacute;", "é"), ("&iacute;", "í"), ("&oacute;", "ó"),
                 ("&uacute;", "ú"), ("&ntilde;", "ñ")]:
        h = h.replace(a, b)
    return [l.strip() for l in h.split("\n") if l.strip()]


def test_el_informe_en_ingles_no_arrastra_frases_en_español(tmp_path):
    """Barrido ancho, no por sondas sueltas.

    Se generan los dos documentos y se comparan sus textos: una linea que
    aparece IGUAL en ambos y tiene pinta de español es una cadena que se quedo
    sin envolver en tr() o cuya clave dejo de coincidir. Es como se encontraron
    las ultimas ~100, y sin esta prueba vuelven a colarse a la primera edicion.
    """
    import re
    es = set(_texto_visible(_informe(tmp_path, "es")))
    en = set(_texto_visible(_informe(tmp_path, "en")))
    marca = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|del|que|con|para|"
                       r"por|una|un|se|es|su|sus|como|cada|sobre|entre|sin|no)\b",
                       re.I)
    # Los apellidos de la bibliografia son nombres propios: no se traducen.
    sospechosas = [t for t in es & en
                   if marca.search(t) and len(t) > 2 and "(2024)" not in t]
    assert not sospechosas, (
        "texto sin traducir en el informe en ingles:\n  "
        + "\n  ".join(sorted(sospechosas)[:15]))


def test_ninguna_traduccion_se_congela_al_importar():
    """Un tr() en el cuerpo del modulo se evalua UNA vez, al importar.

    Con eso, cambiar el idioma y volver a generar el informe deja ese texto en
    el idioma anterior. Ya paso con el nombre del morfotipo. Se comprueba sobre
    el arbol sintactico porque en tiempo de ejecucion no se distingue.
    """
    import ast
    from pathlib import Path as _P

    ruta = _P(__file__).resolve().parents[1] / "polyx" / "core" / "report_html.py"
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    dentro_de_funcion = set()
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in ast.walk(nodo):
                dentro_de_funcion.add(id(sub))
    congeladas = [
        n.lineno for n in ast.walk(arbol)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "tr" and id(n) not in dentro_de_funcion]
    assert not congeladas, (
        f"tr() en el cuerpo del modulo, lineas {congeladas}: se resuelve al "
        f"importar y no sigue al idioma elegido")
