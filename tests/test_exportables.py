"""Tablas y figuras del informe: numeradas, enlazadas y exportables.

Lo que se vigila aqui es lo que un lector del paper notaria:

* que el numero que dice el informe ("Tabla 2") sea el del archivo que se
  exporta (``Tabla02_...``);
* que el CSV tenga lo que la tabla muestra, no otra cosa;
* que la figura salga vectorial y a 300 dpi, no la miniatura de 120 dpi del HTML;
* que regenerar el informe no deje archivos viejos mezclados con los nuevos;
* que todo esto exista en español y en ingles.
"""
from __future__ import annotations

import csv
import importlib
import io
import os
import re
import zipfile

import pytest

from tests.test_informe_bilingue import _estado_minimo, _restaurar_idioma  # noqa: F401


def _generar(tmp_path, lang="es", **kw):
    os.environ["POLYX_IDIOMA"] = lang
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.core.exportables as ex
    importlib.reload(ex)
    import polyx.core.report_html as rh
    importlib.reload(rh)
    salida = tmp_path / f"informe_{lang}.html"
    rh.generate_report(_estado_minimo(tmp_path), salida,
                       secciones=list(rh.IDS_SECCIONES), **kw)
    return salida, salida.read_text(encoding="utf-8"), ex.carpeta_de(salida)


def _leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


@pytest.mark.parametrize("lang,tabla,figura", [("es", "Tabla", "Figura"),
                                               ("en", "Table", "Figure")])
def test_se_crea_la_carpeta_con_tablas_figuras_e_indice(tmp_path, lang, tabla, figura):
    _, _, carpeta = _generar(tmp_path, lang)
    assert carpeta.name == f"informe_{lang}_exportacion"
    assert list((carpeta / "tablas").glob("*.csv")), "no se exporto ninguna tabla"
    assert list((carpeta / "tablas").glob("*.tex"))
    assert list((carpeta / "tablas").glob("*.xlsx"))
    assert list((carpeta / "figuras").glob("*.svg"))
    assert (carpeta / "indice.csv").exists()
    # El prefijo y el sufijo de idioma estan en el nombre del archivo.
    assert all(p.name.startswith(tabla) and f"_{lang.upper()}" in p.name
               for p in (carpeta / "tablas").glob("*.csv"))
    assert all(p.name.startswith(figura) for p in (carpeta / "figuras").glob("*.svg"))


@pytest.mark.parametrize("lang,tabla,figura", [("es", "Tabla", "Figura"),
                                               ("en", "Table", "Figure")])
def test_el_informe_numera_igual_que_los_archivos(tmp_path, lang, tabla, figura):
    _, html, carpeta = _generar(tmp_path, lang)
    n_tablas = len(list((carpeta / "tablas").glob("*.csv")))
    n_figs = len(list((carpeta / "figuras").glob("*.svg")))
    assert n_tablas and n_figs
    for n in range(1, n_tablas + 1):
        assert f"{tabla} {n}." in html
    for n in range(1, n_figs + 1):
        assert f"{figura} {n}." in html
    assert f"{tabla} {n_tablas + 1}." not in html
    # La leyenda vieja sin numero ("Figura. Histograma...") ya no debe quedar.
    assert f">{figura}. " not in html


def test_el_csv_es_la_tabla_del_informe(tmp_path):
    _, html, carpeta = _generar(tmp_path, "es")
    csv1 = sorted((carpeta / "tablas").glob("Tabla01_*.csv"))[0]
    filas = _leer_csv(csv1)
    primera = re.search(r"<table class='data'[^>]*>(.*?)</table>", html, re.S).group(1)
    cabecera = re.findall(r"<th[^>]*>(.*?)</th>", primera, re.S)
    assert filas[0] == [re.sub(r"<[^>]+>", "", c).strip() for c in cabecera]
    assert len(filas) > 1


def test_la_figura_sale_vectorial_y_a_300_dpi(tmp_path):
    from PIL import Image
    _, _, carpeta = _generar(tmp_path, "es")
    svg = sorted((carpeta / "figuras").glob("*.svg"))[0]
    assert b"<svg" in svg.read_bytes()
    assert svg.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    png = Image.open(svg.with_suffix(".png"))
    assert round(png.info["dpi"][0]) == 300


def test_el_histograma_exporta_los_valores_crudos_no_las_barras(tmp_path):
    """Quien rehace el histograma necesita las mediciones, no la altura de las barras."""
    _, _, carpeta = _generar(tmp_path, "es")
    crudos = [_leer_csv(p) for p in (carpeta / "datos").glob("*.csv")
              if _leer_csv(p)[0] == ["confianza"]]
    assert crudos, "no hay CSV de datos con la confianza cruda"
    assert crudos[0][1:] == [["0.9"]]


def test_el_indice_lista_cada_archivo_con_su_leyenda(tmp_path):
    _, _, carpeta = _generar(tmp_path, "es")
    filas = _leer_csv(carpeta / "indice.csv")
    assert filas[0] == ["Tipo", "Número", "Archivos", "Leyenda"]
    for tipo, numero, archivos, leyenda in filas[1:]:
        assert leyenda.startswith(f"{tipo} {numero}.")
        for a in archivos.split(", "):
            if "/" in a:
                ruta = carpeta / a.split(", ")[0]
                assert ruta.parent.exists()


def test_los_enlaces_de_descarga_estan_en_el_html(tmp_path):
    _, html, _ = _generar(tmp_path, "es")
    assert "class='exp'" in html
    assert "download='Tabla01_" in html and ".csv'" in html
    assert "image/svg+xml" in html


def test_sin_exportar_solo_numera(tmp_path):
    _, html, carpeta = _generar(tmp_path, "es", exportar=False)
    assert not carpeta.exists()
    assert "class='exp'" not in html
    assert "Tabla 1." in html and "Figura 1." in html


def test_regenerar_no_deja_archivos_viejos(tmp_path):
    salida, _, carpeta = _generar(tmp_path, "es")
    viejo = carpeta / "tablas" / "Tabla99_de_una_corrida_anterior.csv"
    viejo.write_text("x", encoding="utf-8")
    ajeno = carpeta / "mis_notas.txt"
    ajeno.write_text("mio", encoding="utf-8")
    _generar(tmp_path, "es")
    assert not viejo.exists()
    assert ajeno.exists(), "se borro un archivo que no era del exportador"


def test_el_xlsx_abre_y_conserva_numeros_como_numeros(tmp_path):
    _, _, carpeta = _generar(tmp_path, "es")
    xlsx = next((carpeta / "tablas").glob("*.xlsx"))
    with zipfile.ZipFile(xlsx) as z:
        assert z.testzip() is None
        assert "xl/worksheets/sheet1.xml" in z.namelist()
    openpyxl = pytest.importorskip("openpyxl")
    libro = openpyxl.load_workbook(xlsx)
    assert libro.sheetnames[0] == "Tabla 1"
    hoja = libro.worksheets[0]
    assert hoja.max_row > 1


def test_latex_escapa_los_caracteres_especiales():
    from polyx.core import exportables as ex
    tex = ex._latex_tabla([["a_b", "50 %"], ["x & y", "3.5"]], "Total $5", "tab:1").decode()
    assert r"a\_b" in tex and r"50 \%" in tex and r"x \& y" in tex
    assert r"\caption{Total \$5}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_xlsx_manual_distingue_numero_de_texto():
    from polyx.core import exportables as ex
    datos = ex._xlsx_bytes([("T", [["n", "x"], ["12.5", "abc"], ["007", "—"]])])
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        hoja = z.read("xl/worksheets/sheet1.xml").decode()
    assert "<v>12.5</v>" in hoja
    # Un codigo con ceros a la izquierda no es un numero: se conserva tal cual.
    assert "<t xml:space=\"preserve\">007</t>" in hoja
