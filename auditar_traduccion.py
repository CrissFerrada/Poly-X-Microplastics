"""Mide cuanto de la interfaz esta traducido al ingles y que falta.

Recorre el codigo buscando texto que ve el usuario y lo cruza contra el
diccionario de ``polyx/core/i18n.py``. Sin esto, "traducir el programa" es una
tarea sin fondo visible: aqui se ve el porcentaje y la lista concreta.

Que cuenta como texto de interfaz: literales dentro de las llamadas Qt que
muestran algo (``QLabel``, ``QPushButton``, ``setText``, ``setToolTip``,
``setWindowTitle``, ``addItem``, los mensajes de ``QMessageBox``...). Se
descartan hojas de estilo, nombres de objeto, rutas y claves de diccionario,
que son las que inflaban el recuento a ~2350.

Uso:
    python auditar_traduccion.py            # resumen por modulo
    python auditar_traduccion.py --faltan   # lista lo no traducido
    python auditar_traduccion.py --plantilla pendientes.py
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

# Llamadas cuyo argumento de texto llega a la pantalla.
CONSTRUCTORES = {"QLabel", "QPushButton", "QCheckBox", "QRadioButton",
                 "QGroupBox", "QAction", "QToolButton", "QLineEdit"}
METODOS = {"setText", "setToolTip", "setWindowTitle", "setPlaceholderText",
           "setTitle", "setStatusTip", "addItem", "addItems", "setItemText",
           "setLabelText", "setFormat", "information", "warning", "critical",
           "question", "about", "setHtml"}

# Un literal es texto de interfaz solo si parece una frase, no un identificador.
def es_texto_ui(s: str) -> bool:
    if len(s) < 4 or len(s) > 400:
        return False
    bajo = s.lower()
    if any(t in bajo for t in ("color:", "background", "border", "font-size",
                               "padding", "margin", "qwidget", "qframe",
                               "qpushbutton{", "border-radius")):
        return False
    if s.startswith(("#", "http", "/", "\\", ".", "_")):
        return False
    if s.count(" ") == 0 and "_" in s:      # identificadores tipo modo_rapido
        return False
    return any(c.isalpha() for c in s)


def literales_de(nodo: ast.AST) -> list[str]:
    """Extrae los literales de texto de un nodo, incluida la concatenacion implicita."""
    out = []
    for hijo in ast.walk(nodo):
        if isinstance(hijo, ast.Constant) and isinstance(hijo.value, str):
            out.append(hijo.value)
        elif isinstance(hijo, ast.JoinedStr):
            # f-string: se reconstruye con marcador para poder traducir el molde
            partes = []
            for v in hijo.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    partes.append(v.value)
                else:
                    partes.append("{}")
            out.append("".join(partes))
    return out


def recolectar(archivo: Path) -> tuple[set[str], set[str]]:
    """(envueltas en tr(), crudas). Envolver no puede hacer bajar el total.

    Se separan porque una cadena solo esta lista si cumple las dos cosas:
    envuelta en ``tr()`` **y** presente en el diccionario. Envuelta sin entrada
    sale en espanol; con entrada pero sin envolver, la entrada no se usa nunca.
    """
    try:
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    except SyntaxError:
        return set(), set()
    envueltas: set[str] = set()
    crudas: set[str] = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        f = nodo.func
        nombre = (f.id if isinstance(f, ast.Name)
                  else f.attr if isinstance(f, ast.Attribute) else "")
        if nombre not in CONSTRUCTORES and nombre not in METODOS:
            continue
        for arg in nodo.args:
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                    and arg.func.id == "tr"):
                for s in literales_de(arg):
                    if es_texto_ui(s):
                        envueltas.add(s)
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                # Solo el literal COMPLETO cuenta como unidad traducible. Los
                # trozos sueltos de un f-string o de una concatenacion no lo son:
                # contarlos inflaba el total con fragmentos como " px → " que no
                # se traducen por separado sino como parte de su mensaje.
                if es_texto_ui(arg.value):
                    crudas.add(arg.value)
    return envueltas, crudas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faltan", action="store_true", help="lista lo no traducido")
    ap.add_argument("--plantilla", help="escribe un .py con las entradas vacias")
    args = ap.parse_args()

    from polyx.core.i18n import TRADUCCIONES

    envueltas: dict[str, set[str]] = defaultdict(set)
    crudas: dict[str, set[str]] = defaultdict(set)
    for py in sorted((RAIZ / "polyx").rglob("*.py")):
        if "__pycache__" in py.parts or py.name == "i18n.py":
            continue
        modulo = py.relative_to(RAIZ / "polyx").parts[0]
        if modulo.endswith(".py"):
            modulo = f"({py.stem})"
        e, c = recolectar(py)
        envueltas[modulo] |= e
        crudas[modulo] |= c

    modulos = sorted(set(envueltas) | set(crudas))
    print(f"{'modulo':16}{'cadenas':>9}{'listas':>8}{'sin tr()':>10}"
          f"{'sin traducir':>14}{'%':>6}")
    print("-" * 63)
    total = listas = 0
    for m in modulos:
        e, c = envueltas[m], crudas[m]
        n = len(e | c)
        ok = sum(1 for s in e if s in TRADUCCIONES)
        sin_trad = sum(1 for s in e if s not in TRADUCCIONES)
        total += n
        listas += ok
        print(f"{m:16}{n:>9}{ok:>8}{len(c):>10}{sin_trad:>14}"
              f"{100 * ok / max(n, 1):>5.0f}%")
    print("-" * 63)
    print(f"{'TOTAL':16}{total:>9}{listas:>8}"
          f"{sum(len(c) for c in crudas.values()):>10}"
          f"{sum(1 for m in modulos for s in envueltas[m] if s not in TRADUCCIONES):>14}"
          f"{100 * listas / max(total, 1):>5.0f}%")

    faltantes = sorted({s for m in modulos for s in (envueltas[m] | crudas[m])
                        if s not in TRADUCCIONES})
    if args.faltan:
        print("\nSin traducir:")
        for c in faltantes:
            print(f"  {c!r}")
    if args.plantilla:
        destino = Path(args.plantilla)
        destino.write_text(
            "# Pegar en TRADUCCIONES de polyx/core/i18n.py y completar.\n"
            "PENDIENTES = {\n"
            + "".join(f"    {c!r}: {''!r},\n" for c in faltantes)
            + "}\n", encoding="utf-8")
        print(f"\nPlantilla con {len(faltantes)} entradas en {destino}")


if __name__ == "__main__":
    main()
