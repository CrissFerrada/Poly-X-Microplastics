"""Componentes opcionales: el registro y el diálogo que los instala.

Lo que se vigila:

* que el instalador use el ``python.exe`` del entorno aunque Poly-X corra con
  ``pythonw.exe`` (sin consola no hay salida que mostrar);
* que el comando apunte al archivo de requisitos con versiones fijadas, no a
  un ``pip install paquete`` suelto;
* que el diálogo verifique **después** de instalar, y que un fallo en
  cualquiera de los dos pasos quede a la vista y no como «instalado».

El diálogo se prueba con comandos falsos y rápidos: no instala nada de verdad.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from polyx.core import componentes as C


def test_cada_componente_tiene_su_archivo_de_requisitos():
    for comp in C.COMPONENTES.values():
        assert (C.raiz() / comp.requisitos).exists(), comp.requisitos


def test_el_requisito_de_rfdetr_fija_las_versiones_del_proyecto():
    texto = (C.raiz() / C.componente("rfdetr").requisitos).read_text(encoding="utf-8")
    assert "-c requirements.txt" in texto
    assert "pyarrow<20" in texto


def test_el_comando_instala_desde_el_archivo_y_no_suelto():
    cmd = C.comando_instalar("rfdetr")
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert cmd[-2] == "-r" and cmd[-1].endswith("requirements-rfdetr.txt")


def test_la_verificacion_exige_numpy_1():
    assert "numpy.__version__.startswith('1.')" in C.componente("rfdetr").verificacion


def test_bajo_pythonw_se_usa_el_python_de_al_lado(monkeypatch, tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    (tmp_path / "pythonw.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert C.interprete() == tmp_path / "python.exe"


def test_componente_desconocido_falla_con_mensaje():
    with pytest.raises(ValueError, match="desconocido"):
        C.componente("cellpose-que-no-existe")


# ── Diálogo ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _correr_dialogo(app, monkeypatch, instalar_code, verificar_code):
    from PySide6.QtCore import QEventLoop, QTimer
    from polyx.core import instalador_ui as U
    monkeypatch.setattr(U.C, "comando_instalar",
                        lambda c: [sys.executable, "-c", instalar_code])
    monkeypatch.setattr(U.C, "comando_verificar",
                        lambda c: [sys.executable, "-c", verificar_code])
    dlg = U.DialogoInstalar("rfdetr")
    emitidos = []
    dlg.instalado.connect(emitidos.append)
    dlg._instalar()
    bucle = QEventLoop()
    reloj = QTimer()
    reloj.timeout.connect(lambda: bucle.quit() if not dlg._fase else None)
    reloj.start(50)
    QTimer.singleShot(30000, bucle.quit)
    bucle.exec()
    reloj.stop()
    return dlg, emitidos


def test_instalacion_correcta_verifica_y_avisa(app, monkeypatch):
    dlg, emitidos = _correr_dialogo(app, monkeypatch, "print('instalando')",
                                    "print('verificado')")
    log = dlg._log.toPlainText()
    assert "instalando" in log and "verificado" in log, "faltó un paso"
    assert emitidos == ["rfdetr"]
    assert dlg._estado.text().startswith("✓")
    assert dlg._btn_cerrar.isEnabled()


def test_si_falla_pip_no_se_da_por_instalado(app, monkeypatch):
    dlg, emitidos = _correr_dialogo(app, monkeypatch, "import sys; sys.exit(3)",
                                    "print('no deberia correr')")
    assert emitidos == []
    assert dlg._estado.text().startswith("✗")
    assert "no deberia correr" not in dlg._log.toPlainText()


def test_si_falla_la_verificacion_no_se_da_por_instalado(app, monkeypatch):
    dlg, emitidos = _correr_dialogo(
        app, monkeypatch, "print('ok')",
        "raise SystemExit('numpy 2.4: Ultralytics necesita numpy 1.x')")
    assert emitidos == []
    assert dlg._estado.text().startswith("✗")
    assert "numpy 2.4" in dlg._log.toPlainText()
