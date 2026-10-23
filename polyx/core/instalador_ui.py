"""Instalar componentes opcionales desde la interfaz, sin abrir una consola.

``DialogoInstalar`` corre pip con el Python del propio programa, muestra lo que
va pasando y, al terminar, verifica en un proceso nuevo que el componente carga
y que no se movió nada de lo que necesita el resto (numpy 1.x para Ultralytics).

``SeccionComponentes`` es la lista para Preferencias: un renglón por cada
entrada de ``core.componentes.COMPONENTES``, con su estado y su botón.

La instalación corre en un ``QProcess`` y no en un hilo de Python: pip no es
una biblioteca que se pueda llamar desde dentro, y si fallara a medias dejaría
el intérprete del programa en un estado desconocido.
"""
from __future__ import annotations

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from . import componentes as C
from . import iconos
from . import theme as T
from .i18n import tr


class DialogoInstalar(QDialog):
    """Instala un componente opcional y verifica el resultado."""

    instalado = Signal(str)   # clave del componente, al terminar bien

    def __init__(self, clave: str, parent=None):
        super().__init__(parent)
        self._c = C.componente(clave)
        self._proc: QProcess | None = None
        self._fase = ""          # "instalar" | "verificar" | ""
        self.setWindowTitle(tr("Instalar {}").format(self._c.nombre))
        self.setMinimumSize(640, 460)
        self.setStyleSheet(T.GLOBAL_QSS + f"QDialog {{ background: {T.BG_SOFT}; }}")

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(22, 20, 22, 18)
        raiz.setSpacing(8)

        titulo = QLabel(self._c.nombre)
        titulo.setStyleSheet(f"color: {T.INK}; font-size: 13pt; font-weight: 650;")
        raiz.addWidget(titulo)

        desc = QLabel(tr(self._c.descripcion) + "<br>"
                      + tr("Se descargan {}. Tu instalación actual no cambia: las "
                           "versiones quedan atadas a las de Poly-X.").format(tr(self._c.tamano)))
        desc.setWordWrap(True)
        desc.setTextFormat(Qt.RichText)
        desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt;")
        raiz.addWidget(desc)

        self._barra = QProgressBar()
        self._barra.setRange(0, 1)
        self._barra.setValue(0)
        self._barra.setTextVisible(False)
        self._barra.setFixedHeight(6)
        raiz.addWidget(self._barra)

        self._estado = QLabel(tr("Listo para instalar."))
        self._estado.setWordWrap(True)
        self._estado.setStyleSheet(f"color: {T.INK2}; font-size: 10pt;")
        raiz.addWidget(self._estado)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "background: #0d1117; color: #c9d1d9; font-family: Consolas, monospace; "
            "font-size: 9pt; border-radius: 6px;")
        raiz.addWidget(self._log, 1)

        pie = QHBoxLayout()
        pie.addStretch(1)
        self._btn_cerrar = QPushButton(tr("Cerrar"))
        self._btn_cerrar.setCursor(Qt.PointingHandCursor)
        self._btn_cerrar.clicked.connect(self.close)
        pie.addWidget(self._btn_cerrar)
        self._btn_instalar = QPushButton(tr("Instalar"))
        self._btn_instalar.setObjectName("primary")
        self._btn_instalar.setIcon(iconos.icono("paquete", 15, T.ON_ACCENT))
        self._btn_instalar.setCursor(Qt.PointingHandCursor)
        self._btn_instalar.clicked.connect(self._instalar)
        pie.addWidget(self._btn_instalar)
        raiz.addLayout(pie)

        if C.instalado(clave):
            self._estado.setText("✓ " + tr("{} ya está instalado. Puedes reinstalarlo "
                                           "si algo falla.").format(self._c.nombre))
            self._btn_instalar.setText(tr("Reinstalar"))

    # ── Procesos ────────────────────────────────────────────────────
    def _correr(self, comando: list[str], fase: str) -> None:
        self._fase = fase
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(C.raiz()))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._leer)
        self._proc.finished.connect(self._terminado)
        self._proc.errorOccurred.connect(self._error_proceso)
        self._escribir("$ " + " ".join(comando))
        self._proc.start(comando[0], comando[1:])

    def _instalar(self) -> None:
        self._btn_instalar.setEnabled(False)
        self._btn_cerrar.setEnabled(False)
        self._barra.setRange(0, 0)       # indeterminada: pip no informa un total
        self._estado.setText(tr("Instalando… puede tardar varios minutos. No cierres "
                                "Poly-X."))
        self._correr(C.comando_instalar(self._c.clave), "instalar")

    def _leer(self) -> None:
        if self._proc is None:
            return
        texto = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for linea in texto.replace("\r", "\n").splitlines():
            if linea.strip():
                self._escribir(linea)

    def _escribir(self, linea: str) -> None:
        self._log.appendPlainText(linea)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _error_proceso(self, _err) -> None:
        # FailedToStart no emite finished: sin esto el diálogo quedaba bloqueado.
        if self._proc is not None and self._proc.state() == QProcess.NotRunning:
            self._fallo(tr("No se pudo ejecutar Python: {}").format(self._proc.errorString()))

    def _terminado(self, codigo: int, _estado) -> None:
        if codigo != 0:
            if self._fase == "instalar":
                self._fallo(tr("La instalación falló (código {}). Revisa el registro "
                               "de arriba.").format(codigo))
            else:
                self._fallo(tr("Se instaló, pero la verificación falló. No entrenes "
                               "hasta revisarlo: copia el registro y avisa."))
            return
        if self._fase == "instalar":
            self._estado.setText(tr("Verificando que todo siga en orden…"))
            self._correr(C.comando_verificar(self._c.clave), "verificar")
            return
        self._fase = ""
        self._barra.setRange(0, 1)
        self._barra.setValue(1)
        self._btn_cerrar.setEnabled(True)
        self._btn_instalar.setText(tr("Reinstalar"))
        self._btn_instalar.setEnabled(True)
        self._estado.setText("✓ " + tr("{} quedó instalado. Cierra y vuelve a abrir este "
                                       "módulo para usarlo.").format(self._c.nombre))
        self.instalado.emit(self._c.clave)

    def _fallo(self, mensaje: str) -> None:
        self._fase = ""
        self._barra.setRange(0, 1)
        self._barra.setValue(0)
        self._btn_cerrar.setEnabled(True)
        self._btn_instalar.setEnabled(True)
        self._estado.setText("✗ " + mensaje)
        self._escribir("✗ " + mensaje)

    def closeEvent(self, ev) -> None:
        # Cortar pip a mitad puede dejar paquetes a medio escribir.
        if self._fase:
            r = QMessageBox.question(
                self, tr("Instalación en curso"),
                tr("La instalación todavía está corriendo. Si la detienes, el "
                   "componente puede quedar a medio instalar.\n\n¿Detenerla igual?"))
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            if self._proc is not None:
                self._proc.kill()
                self._proc.waitForFinished(3000)
        super().closeEvent(ev)


def ofrecer_instalacion(clave: str, parent=None) -> bool:
    """Abre el diálogo de instalación. Devuelve True si terminó instalado."""
    dlg = DialogoInstalar(clave, parent)
    hecho = {"ok": False}
    dlg.instalado.connect(lambda _c: hecho.update(ok=True))
    dlg.exec()
    return hecho["ok"]


class SeccionComponentes(QWidget):
    """Un renglón por componente opcional, con su estado y su botón."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._filas: dict[str, tuple[QLabel, QPushButton]] = {}
        for clave, comp in C.COMPONENTES.items():
            fila = QHBoxLayout()
            fila.setSpacing(10)
            texto = QLabel()
            texto.setWordWrap(True)
            texto.setTextFormat(Qt.RichText)
            texto.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; background: transparent;")
            fila.addWidget(texto, 1)
            boton = QPushButton()
            boton.setCursor(Qt.PointingHandCursor)
            boton.setIcon(iconos.icono("paquete", 14, T.INK2))
            boton.clicked.connect(lambda _=False, c=clave: self._abrir(c))
            fila.addWidget(boton)
            lay.addLayout(fila)
            self._filas[clave] = (texto, boton)
        self.refrescar()

    def refrescar(self) -> None:
        for clave, (texto, boton) in self._filas.items():
            comp = C.componente(clave)
            if C.instalado(clave):
                estado = f"<span style='color:{T.OK_TX}'>✓ {tr('instalado')}</span>"
                boton.setText(tr("Reinstalar…"))
            else:
                estado = f"<span style='color:{T.INK3}'>{tr('no instalado')}</span>"
                boton.setText(tr("Instalar…"))
            texto.setText(f"<b>{comp.nombre}</b> · {estado}<br>"
                          f"<span style='color:{T.INK3}'>{tr(comp.descripcion)}</span>")

    def _abrir(self, clave: str) -> None:
        ofrecer_instalacion(clave, self)
        self.refrescar()
