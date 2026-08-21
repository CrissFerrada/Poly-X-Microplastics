"""Mide la particula de verdad, no su caja.

La caja de un detector son cuatro numeros y no describe la forma. Para una
particula alargada tumbada en diagonal la caja esta casi vacia: una fibra de
2000x20 um a 45 grados tiene una caja de 1414x1414, cuya area es 50 veces la de
la fibra. Y lo peor no es la magnitud sino que **la misma particula rotada da
otro numero**, con lo que la medida no es una medida.

Aqui se segmenta dentro de la caja -- que ya dice donde mirar, asi que no hace
falta anotar mascaras ni reentrenar nada -- y las magnitudes salen del contorno.

Sobre el largo de una particula curva. Una fibra doblada mide mas que la
distancia entre sus extremos, asi que la cuerda la subestima. Se calcula por dos
vias que se apoyan en supuestos distintos:

  * MODELO DE RECTANGULO. Suponiendo ancho constante, area y perimetro dan el
    largo y el ancho resolviendo A = L*W y P = 2(L+W):

        L = (P + sqrt(P^2 - 16A)) / 4        W = (P - sqrt(P^2 - 16A)) / 4

    Lo valioso es que **doblar una fibra no cambia ni su area ni su perimetro**,
    de modo que esto da el largo verdadero de una fibra curva sin modelar la
    curva. Solo vale si el discriminante es positivo: para una particula compacta
    P^2 < 16A (un circulo da 4*pi*r^2*(pi-4) < 0) y entonces no hay rectangulo
    que la describa.

  * CUERDA. El lado mayor del rectangulo de area minima (``minAreaRect``), que
    es la extension en linea recta y no depende de la orientacion, al contrario
    que la caja del detector.

El cociente entre las dos es una medida de cuanto se curva la particula, y sale
gratis: en una recta coinciden, y cuanto mas doblada este mas grande es el
cociente. Se reporta para poder decir en el paper cuantas particulas eran
curvas, en vez de suponerlo.

Solo se usan cv2, numpy y scipy, que ya son dependencias: anadir skimage u
opencv-contrib es justo lo que rompe la instalacion en otra maquina.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Morfotipos. Umbral en 3 sobre el aspecto REAL (no el de la caja): es el corte
# que se usa habitualmente para separar fibra de fragmento.
FIBRA = "fibra"
FRAGMENTO = "fragmento"
INDETERMINADO = "indeterminado"

ASPECTO_FIBRA = 3.0

# Por encima de esto la particula se considera curva y el largo de la cuerda
# deja de servir como talla. 1.15 = el contorno recorre un 15% mas que la recta.
CURVA_DESDE = 1.15


@dataclass
class Morfologia:
    """Forma y talla de una particula, medidas sobre su mascara."""
    ok: bool = False
    # px
    area_px: float = 0.0
    perimetro_px: float = 0.0
    largo_px: float = 0.0        # por el modelo de rectangulo si aplica
    ancho_px: float = 0.0
    cuerda_px: float = 0.0       # extension recta (minAreaRect)
    # um, si hay calibracion
    area_um2: Optional[float] = None
    largo_um: Optional[float] = None
    ancho_um: Optional[float] = None
    # adimensionales
    aspecto: float = 0.0
    circularidad: float = 0.0
    curvatura: float = 1.0       # largo / cuerda; 1.0 = recta
    llenado: float = 0.0         # area de la particula / area de su caja
    morfotipo: str = INDETERMINADO
    metodo: str = ""             # de donde salio el largo
    aviso: str = ""
    # La medida sale, pero pide un vistazo humano. Se marca en vez de descartar:
    # descartar en silencio pierde particulas reales del conteo, y dar el numero
    # sin avisar mete en la tabla tallas que no se sostienen.
    revisar: bool = False

    @property
    def curva(self) -> bool:
        return self.curvatura >= CURVA_DESDE


def _recorte(bgr: np.ndarray, x1, y1, x2, y2, margen: int):
    """Recorta la caja con margen, y devuelve tambien el desplazamiento.

    El margen no es cosmetico: para separar la particula del fondo por umbral
    hace falta que en el recorte haya fondo. Una caja ajustada al borde de la
    particula es casi toda particula, y Otsu partiria la particula en dos en vez
    de separarla del fondo.
    """
    H, W = bgr.shape[:2]
    ax = max(0, int(x1) - margen)
    ay = max(0, int(y1) - margen)
    bx = min(W, int(np.ceil(x2)) + margen)
    by = min(H, int(np.ceil(y2)) + margen)
    if bx - ax < 3 or by - ay < 3:
        return None, 0, 0
    return bgr[ay:by, ax:bx], ax, ay


def segmentar(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              margen: Optional[int] = None) -> Optional[np.ndarray]:
    """Mascara de la particula dentro de la caja. None si no se pudo aislar.

    Bajo Nile Red la particula emite y el fondo del filtro queda oscuro, asi que
    se umbraliza el canal de mayor contraste con Otsu. Se toma la componente
    conexa que cubre el centro de la caja: dentro del recorte pueden caer trozos
    de particulas vecinas, y la que interesa es la que el detector senalo.
    """
    if bgr is None or bgr.size == 0:
        return None
    ancho_caja = max(1.0, x2 - x1)
    alto_caja = max(1.0, y2 - y1)
    if margen is None:
        # Proporcional, con minimo: en una caja de 30 px un margen fijo de 20
        # meteria mas vecinos que fondo util.
        margen = int(max(3, round(0.35 * min(ancho_caja, alto_caja))))

    sub, ox, oy = _recorte(bgr, x1, y1, x2, y2, margen)
    if sub is None:
        return None

    # El canal mas contrastado, no siempre el gris: PP y LDPE emiten en amarillo
    # (R y G altos, B bajo) y PET en rojo, de modo que el maximo por pixel separa
    # mejor del fondo que la media ponderada que usa BGR2GRAY.
    canal = sub.max(axis=2)
    canal = cv2.GaussianBlur(canal, (3, 3), 0)
    _, mascara = cv2.threshold(canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Quita el ruido de sal sin cerrar el hueco de una particula con forma de U.
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8), iterations=1)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return None
    # Centro de la caja en coordenadas del recorte.
    cx = int(round((x1 + x2) / 2 - ox))
    cy = int(round((y1 + y2) / 2 - oy))
    cx = int(np.clip(cx, 0, etiquetas.shape[1] - 1))
    cy = int(np.clip(cy, 0, etiquetas.shape[0] - 1))
    idx = etiquetas[cy, cx]
    if idx == 0:
        # El centro cayo en fondo -- particula anular o caja algo descentrada.
        # Se usa la componente mayor que toque la caja.
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 4:
        return None
    mascara = (etiquetas == idx).astype(np.uint8) * 255

    # Recortar a la caja. El margen existe para que Otsu tenga fondo con que
    # comparar, no para medir en el: sin este recorte, una particula pegada a
    # otra forma con ella una sola componente conexa y se mide el conjunto. Se
    # observo en placas reales una mascara 1.71 veces mas ancha que su caja, y
    # de ahi salian tallas de 8.9 mm que ninguna caja de 97 px puede contener.
    #
    # Se deja una holgura: el detector suele ajustar la caja algo por dentro del
    # borde real, y cortar a ras amputaria la punta de la particula.
    holgura = max(2, int(round(0.10 * min(ancho_caja, alto_caja))))
    hx1 = max(0, int(x1) - ox - holgura)
    hy1 = max(0, int(y1) - oy - holgura)
    hx2 = min(mascara.shape[1], int(np.ceil(x2)) - ox + holgura)
    hy2 = min(mascara.shape[0], int(np.ceil(y2)) - oy + holgura)
    fuera = np.ones_like(mascara, dtype=bool)
    fuera[hy1:hy2, hx1:hx2] = False
    mascara[fuera] = 0
    if not mascara.any():
        return None
    return mascara


def medir(mascara: np.ndarray, um_por_px: Optional[float] = None) -> Morfologia:
    """Talla y forma a partir de la mascara de una particula."""
    m = Morfologia()
    if mascara is None or not mascara.any():
        m.aviso = "sin mascara"
        return m

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contornos:
        m.aviso = "sin contorno"
        return m
    c = max(contornos, key=cv2.contourArea)

    # El area se cuenta sobre la mascara y no con contourArea: la formula del
    # poligono subestima en particulas de pocos pixeles, que son la mayoria aqui.
    area = float(np.count_nonzero(mascara))
    # El contorno se recorre pixel a pixel, SIN suavizar. Se probo suavizarlo
    # con approxPolyDP para quitar la escalera de digitalizacion, contra formas
    # sinteticas de talla conocida, y empeora: con epsilon 1 px el error mediano
    # sube de 1.0% a 2.4% y el peor de 3.1% a 7.8%, porque recorta los recodos
    # reales de la particula junto con la escalera. Tambien se probo estimar el
    # largo como area/grosor con la transformada de distancia, inmune al
    # perimetro, y es peor todavia: 26.6% de error mediano.
    perim = float(cv2.arcLength(c, True))
    if area < 4 or perim <= 0:
        m.aviso = "particula demasiado pequena para medir su forma"
        return m

    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    cuerda = float(max(rw, rh))
    grosor_recto = float(min(rw, rh))

    # Modelo de rectangulo: valido solo si existe un rectangulo con esa area y
    # ese perimetro. En una particula compacta el discriminante es negativo.
    disc = perim * perim - 16.0 * area
    if disc > 0:
        raiz = float(np.sqrt(disc))
        largo = (perim + raiz) / 4.0
        ancho = (perim - raiz) / 4.0
        metodo = "rectangulo (area+perimetro)"
    else:
        # Compacta: el largo que tiene sentido es su extension recta.
        largo = cuerda
        ancho = grosor_recto
        metodo = "cuerda (particula compacta)"

    # ¿Es de verdad una cinta? El modelo supone ancho constante, y a un grumo de
    # borde rugoso le atribuye un perimetro grande, que traduce en "cinta larga y
    # fina". Distinguirlos: una fibra enrollada es delgada en TODO su recorrido,
    # mientras que un grumo es grueso por dentro. El radio del mayor circulo que
    # cabe dentro de la mascara mide exactamente eso, y no depende del perimetro.
    #
    # Sin esta comprobacion, una particula real de este material daba 8595 um de
    # largo dentro de una mascara cuya diagonal era 3900: el modelo la describia
    # como una cinta de 260 px enrollada, y era un grumo.
    dt = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    grosor_real = 2.0 * float(dt.max())
    curvatura = largo / cuerda if largo > 0 and cuerda > 0 else 1.0

    if disc > 0 and ancho > 0 and grosor_real > 2.0 * ancho:
        # El modelo dice que es el doble de fina de lo que realmente es en su
        # punto mas grueso: no es una cinta.
        largo, ancho = cuerda, grosor_recto
        metodo = "cuerda (no es una cinta: el interior es grueso)"
        m.aviso = (f"el modelo de rectangulo daba un ancho de {ancho:.0f} px pero la "
                   f"particula mide {grosor_real:.0f} px en su punto mas grueso")
        curvatura = 1.0
    elif curvatura > 4.0:
        # Curvatura implausible aunque el grosor cuadre.
        largo, ancho = cuerda, grosor_recto
        metodo = "cuerda (el modelo de rectangulo se disparo)"
        m.aviso = "contorno irregular; el modelo de rectangulo no era fiable"
        curvatura = 1.0

    m.ok = True
    m.area_px = area
    m.perimetro_px = perim
    m.largo_px = largo
    m.ancho_px = max(ancho, 0.0)
    m.cuerda_px = cuerda
    m.aspecto = largo / ancho if ancho > 0 else 0.0
    m.circularidad = float(4.0 * np.pi * area / (perim * perim))
    m.curvatura = curvatura
    m.llenado = area / (rw * rh) if rw * rh > 0 else 0.0
    m.metodo = metodo
    m.morfotipo = FIBRA if m.aspecto >= ASPECTO_FIBRA else FRAGMENTO

    if um_por_px and um_por_px > 0:
        m.area_um2 = area * um_por_px * um_por_px
        m.largo_um = largo * um_por_px
        m.ancho_um = m.ancho_px * um_por_px
    return m


def medir_deteccion(bgr: np.ndarray, det, um_por_px: Optional[float] = None,
                    margen: Optional[int] = None) -> Morfologia:
    """Segmenta la caja de una ``Detection`` y la mide. Nunca lanza.

    Un fallo de segmentacion en una particula no puede tumbar el lote: se
    devuelve una Morfologia con ``ok`` en False y quien llama decide si cae al
    tamano de la caja.
    """
    try:
        mascara = segmentar(bgr, det.x1, det.y1, det.x2, det.y2, margen)
        if mascara is None:
            m = Morfologia()
            m.aviso = "no se pudo separar la particula del fondo"
            return m
        m = medir(mascara, um_por_px)
        # Una particula solo puede ser mas larga que la diagonal de su caja si
        # esta muy enrollada. Pasado 1.5x, lo mas frecuente en este material no
        # es una fibra enrollada sino DOS particulas que se tocan y que la
        # componente conexa unio en una: se comprobo a ojo sobre las mayores del
        # lote. La medida se entrega marcada para revisar, no se descarta.
        diagonal = float(np.hypot(det.x2 - det.x1, det.y2 - det.y1))
        if m.ok and diagonal > 0 and m.largo_px > 1.5 * diagonal:
            m.revisar = True
            m.aviso = (m.aviso + " | " if m.aviso else "") + (
                f"largo {m.largo_px / diagonal:.1f}x la diagonal de su caja: "
                f"comprobar que no sean dos particulas pegadas")
        return m
    except cv2.error as e:
        m = Morfologia()
        m.aviso = f"error de OpenCV al segmentar: {e}"
        return m


def aplicar_a_deteccion(det, bgr: np.ndarray,
                        um_por_px: Optional[float] = None) -> bool:
    """Mide la particula y vuelca el resultado en la ``Detection``.

    Devuelve si se pudo medir. Cuando no, los campos de forma quedan en None y
    quien llama debe caer al tamano de la caja: es peor perder la particula del
    conteo que reportarla con una talla aproximada y decirlo.

    ``diam_um`` pasa a ser el diametro del circulo de igual area **real**, no el
    de la caja. Es la misma magnitud de siempre pero bien calculada, de modo que
    los filtros por tamano y el histograma del informe siguen funcionando sin
    tocarlos, ahora sobre el numero correcto.
    """
    m = medir_deteccion(bgr, det, um_por_px)
    if not m.ok:
        return False
    det.aspecto = m.aspecto
    det.curvatura = m.curvatura
    det.morfotipo = m.morfotipo
    if um_por_px and um_por_px > 0:
        det.area_um2 = m.area_um2
        det.largo_um = m.largo_um
        det.ancho_um = m.ancho_um
        det.diam_um = 2.0 * float(np.sqrt(m.area_um2 / np.pi)) if m.area_um2 else None
    return True
