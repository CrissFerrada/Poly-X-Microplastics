"""Mide la particula de verdad, no su caja.

La caja de un detector son cuatro numeros y no describe la forma. Para una
particula alargada tumbada en diagonal la caja esta casi vacia: una fibra de
2000x20 um a 45 grados tiene una caja de 1414x1414, cuya area es 50 veces la de
la fibra. Y lo peor no es la magnitud sino que **la misma particula rotada da
otro numero**, con lo que la medida no es una medida.

Aqui se segmenta dentro de la caja -- que ya dice donde mirar, asi que no hace
falta anotar mascaras ni reentrenar nada -- y las magnitudes salen de la mascara.

COMO SE MIDE EL LARGO
---------------------
Se calculan dos cotas y se reporta la mayor:

  * FERET MAXIMO. La mayor distancia entre dos puntos del borde, sobre la
    envolvente convexa. Es la medida correcta del largo aparente y de una fibra
    RECTA, y no la afecta un borde dentado. Pero en una fibra curva da la
    cuerda: en un arco de 180 grados se queda un 35% corto.

  * DIAMETRO GEODESICO. El camino mas largo que cabe DENTRO de la particula,
    por doble propagacion. Al no poder salirse de la mascara, rodea la curva y
    devuelve su longitud. En una particula compacta el camino mas largo es el
    recto y coincide con Feret.

Se toma el mayor de los dos porque en una forma conexa el geodesico siempre es
mayor o igual que Feret -- en una convexa coinciden --, de modo que si Feret
sale mayor es solo error de discretizacion y el maximo es la cota mas ajustada.

QUE SE DESCARTO, Y POR QUE
--------------------------
El rectangulo de igual area y perimetro, L = (P + sqrt(P^2-16A))/4, se uso como
talla y NO sirve para eso, aunque se conserva como descriptor:

  * Depende del perimetro, y un borde dentado lo infla. Medido sobre una recta
    con dientes de sierra de talla conocida: +22.5%.
  * Solo existe si P^2 >= 16A. Un circulo da P^2 = 4*pi*A < 16A y no tiene
    rectangulo equivalente.
  * No es "el largo de la particula" sino el de un rectangulo con su misma area
    y su mismo perimetro, que es otra cosa.

Se conserva en ``largo_rect_eq`` porque comparado con Feret y con el geodesico
informa de lo irregular que es la particula: si es mucho mayor que los otros
dos, el borde esta dentado.

Contra formas sinteticas de talla conocida, el largo reportado da 3.7% de error
mediano y 4.7% en el peor caso, salvo una excepcion documentada abajo. El
modelo de rectangulo daba 1.2% mediano pero 22.5% en el peor caso, y aqui
importa mas no fallar feo que afinar en el caso bueno.

LIMITE CONOCIDO
---------------
En una fibra muy enroscada el camino geodesico corta por dentro en cada codo y
subestima: en una fibra en ese de curvas cerradas se queda un 19% corto. Por eso
se reporta tambien ``largo_rect_eq``, que en ese caso concreto acierta mas
(-5%): si las dos cifras discrepan mucho, la particula pide un vistazo.

Solo se usan cv2 y numpy. Ni scipy -- que no es dependencia declarada y solo
llega de rebote con ultralytics -- ni skimage ni opencv-contrib: anadir
dependencias es justo lo que rompe la instalacion en otra maquina.
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

# El diametro geodesico solo se usa si la particula es al menos asi de delgada,
# medida como largo/grosor. En una particula gruesa cualquier concavidad hace
# que el camino geodesico la RODEE en vez de atravesarla, y entonces infla: se
# vio en una particula real de 44 px de extension y 22 de grosor a la que el
# geodesico le daba 73 px por bordear una muesca.
#
# El umbral no es delicado: sobre formas de talla conocida, los grumos quedan en
# 1.05 y 1.64 de delgadez y la fibra mas gorda en 6.7, asi que cualquier valor
# entre 3 y 6 separa igual. Con el, el error mediano del largo baja de 3.1% a
# 0.8% -- un circulo pasa de 104 a 100 exacto -- sin tocar ninguna fibra.
DELGADEZ_PARA_GEODESICO = 4.0

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
    largo_px: float = 0.0        # la talla que se reporta: max(feret, geodesico)
    ancho_px: float = 0.0
    cuerda_px: float = 0.0       # extension recta (minAreaRect)
    feret_px: float = 0.0        # mayor distancia entre dos puntos del borde
    geodesico_px: float = 0.0    # camino mas largo DENTRO de la particula
    # Largo del rectangulo de igual area y perimetro. NO es "el largo de la
    # particula": es el de un rectangulo equivalente, y un borde dentado infla
    # el perimetro y con el esta cifra. Se conserva porque comparado con feret y
    # geodesico informa de lo irregular o filamentosa que es la particula.
    largo_rect_eq_px: float = 0.0
    # um, si hay calibracion
    area_um2: Optional[float] = None
    largo_um: Optional[float] = None
    ancho_um: Optional[float] = None
    feret_um: Optional[float] = None
    geodesico_um: Optional[float] = None
    largo_rect_eq_um: Optional[float] = None
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


def feret_maximo(contorno: np.ndarray) -> float:
    """Diametro de Feret maximo: la mayor distancia entre dos puntos del borde.

    Se calcula sobre la envolvente convexa, no sobre el contorno entero: el par
    mas separado siempre esta en la envolvente, y esta suele tener decenas de
    puntos frente a los miles del contorno, asi que la busqueda exhaustiva es
    barata y exacta.

    Es la medida correcta para el "largo maximo aparente" y para una fibra
    RECTA. Para una fibra curva da la cuerda, que subestima su longitud.
    """
    if contorno is None or len(contorno) < 2:
        return 0.0
    env = cv2.convexHull(contorno).reshape(-1, 2).astype(float)
    if len(env) < 2:
        return 0.0
    d = env[:, None, :] - env[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def _propagar_geodesico(dentro: np.ndarray, semilla: tuple) -> np.ndarray:
    """Distancia geodesica desde una semilla, sin salir de la mascara.

    Propagacion por minimos sucesivos sobre los 8 vecinos, con peso 1 en recto
    y raiz de 2 en diagonal. Vectorizado: cada iteracion son ocho
    desplazamientos de la matriz, y hacen falta tantas como pixeles tenga el
    camino mas largo. Sobre recortes de particula -- decenas o pocos cientos de
    pixeles de lado -- eso es inmediato.

    Se hace asi y no con scipy.sparse.csgraph porque scipy no es una dependencia
    declarada del proyecto: llega de rebote con ultralytics, y apoyarse en eso
    romperia la instalacion el dia que ultralytics deje de traerlo.
    """
    INF = np.inf
    d = np.full(dentro.shape, INF, dtype=np.float32)
    d[semilla] = 0.0
    # Pesos optimos del chamfer 3x3, no 1 y raiz de 2. Con los pesos ingenuos el
    # camino en escalera sobreestima la distancia euclidea hasta un 8% -- medido
    # sobre un circulo de diametro conocido, daba 108 en vez de 100 --. Estos
    # dos minimizan ese error maximo y lo dejan en torno al 1.4%.
    UNO = np.float32(0.95509)
    R2 = np.float32(1.36930)
    # Cota de seguridad: ningun camino simple puede tener mas pasos que pixeles.
    tope = int(dentro.sum()) + 2
    for _ in range(tope):
        p = np.full_like(d, INF)
        # Rectos
        p[1:, :] = np.minimum(p[1:, :], d[:-1, :] + UNO)
        p[:-1, :] = np.minimum(p[:-1, :], d[1:, :] + UNO)
        p[:, 1:] = np.minimum(p[:, 1:], d[:, :-1] + UNO)
        p[:, :-1] = np.minimum(p[:, :-1], d[:, 1:] + UNO)
        # Diagonales
        p[1:, 1:] = np.minimum(p[1:, 1:], d[:-1, :-1] + R2)
        p[1:, :-1] = np.minimum(p[1:, :-1], d[:-1, 1:] + R2)
        p[:-1, 1:] = np.minimum(p[:-1, 1:], d[1:, :-1] + R2)
        p[:-1, :-1] = np.minimum(p[:-1, :-1], d[1:, 1:] + R2)
        p[~dentro] = INF
        nuevo = np.minimum(d, p)
        # Basta comparar con el paso anterior: la propagacion es monotona
        # decreciente, asi que si nada bajo, ya converge.
        if np.array_equal(nuevo, d):
            break
        d = nuevo
    return d


def largo_geodesico(mascara: np.ndarray) -> float:
    """Camino mas largo que cabe DENTRO de la particula, siguiendo su forma.

    Es el diametro geodesico. Hace lo correcto en los dos casos que importan y
    sin cambiar de formula:

      * Fibra curva: el camino no puede salirse de la particula, asi que rodea
        la curva y devuelve su longitud, no la cuerda entre los extremos.
      * Particula compacta: el camino mas largo es el recto, y coincide con el
        diametro de Feret maximo.

    Se obtiene con la tecnica de las dos propagaciones, la misma que da el
    diametro de un arbol: desde un punto cualquiera se busca el mas lejano, y
    desde ese se vuelve a propagar; el maximo de esa segunda propagacion es el
    diametro. No depende del perimetro, de modo que un borde dentado no lo
    infla como si hace el modelo de rectangulo.
    """
    idx = np.argwhere(mascara > 0)
    if len(idx) < 2:
        return 0.0
    # Recortar al minimo rectangulo que contiene la particula. El coste de la
    # propagacion va con el area del lienzo, y una particula de 30 px dentro de
    # una placa de 4096 costaria lo mismo que la placa entera.
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    dentro = (mascara[y0:y1_, x0:x1_] > 0)
    idx = np.argwhere(dentro)

    d1 = _propagar_geodesico(dentro, tuple(idx[0]))
    finitos = np.isfinite(d1)
    if not finitos.any():
        return 0.0
    a = np.unravel_index(np.argmax(np.where(finitos, d1, -1)), d1.shape)

    d2 = _propagar_geodesico(dentro, a)
    finitos2 = np.isfinite(d2)
    return float(d2[finitos2].max()) if finitos2.any() else 0.0


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
    # Rectangulo de igual area y perimetro. Descriptor, NO la talla: solo existe
    # si P^2 >= 16A, que es tanto como decir que la particula es lo bastante
    # alargada -- un circulo da P^2 = 4*pi*A < 16A y no tiene rectangulo
    # equivalente. Comparado con feret y geodesico dice si la particula es
    # irregular: si es mucho mayor que ellos, el borde esta dentado.
    disc = perim * perim - 16.0 * area
    largo_rect_eq = (perim + float(np.sqrt(disc))) / 4.0 if disc > 0 else 0.0

    # ── El largo que se reporta ──
    # Feret maximo y diametro geodesico, y se toma el mayor. Para una forma
    # conexa el geodesico SIEMPRE es mayor o igual que Feret -- en una convexa
    # coinciden --, asi que cuando Feret sale mayor es solo error de
    # discretizacion del chamfer, y quedarse con el mayor toma la cota mas
    # ajustada de las dos.
    #
    # Se prefiere esto al modelo de rectangulo, medido contra formas de talla
    # conocida, porque el modelo depende del perimetro y un borde dentado lo
    # infla: en una recta con dientes de sierra daba +22.5%. max(Feret,
    # geodesico) tiene un error mediano algo mayor (3.6% frente a 1.2%) pero su
    # peor caso es 4.7% en vez de 22.5%, y aqui importa mas no fallar feo.
    feret = feret_maximo(c)
    geo = largo_geodesico(mascara)

    # El grosor decide cual de los dos vale. Solo en una particula delgada el
    # camino geodesico sigue su forma; en una gruesa rodea las concavidades y
    # devuelve un numero mayor que su extension real.
    dt = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    grosor = 2.0 * float(dt.max())
    delgadez = geo / grosor if grosor > 0 else 0.0
    if delgadez >= DELGADEZ_PARA_GEODESICO and geo >= feret:
        largo, metodo = geo, "geodesico"
    else:
        largo, metodo = feret, "Feret maximo"

    # El ancho es ese mismo grosor maximo inscrito: el diametro del mayor
    # circulo que cabe dentro de la particula.
    #
    # No se usa area/largo, que seria el ancho medio de una cinta, porque en una
    # particula compacta no da su diametro: un disco tiene A/L = pi*d/4 = 0.785d,
    # con lo que un circulo perfecto salia con aspecto 1.27 en vez de 1.0 y la
    # clasificacion fibra/fragmento arrancaba sesgada.
    ancho = grosor

    # Cuanto se aparta de una recta. Con el geodesico esto ya no depende del
    # perimetro: es cuanto mas largo es el camino por dentro que la cuerda.
    curvatura = largo / cuerda if largo > 0 and cuerda > 0 else 1.0

    m.ok = True
    m.area_px = area
    m.perimetro_px = perim
    m.largo_px = largo
    m.ancho_px = max(ancho, 0.0)
    m.cuerda_px = cuerda
    m.feret_px = feret
    m.geodesico_px = geo
    m.largo_rect_eq_px = largo_rect_eq
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
        m.feret_um = feret * um_por_px
        m.geodesico_um = geo * um_por_px
        m.largo_rect_eq_um = largo_rect_eq * um_por_px if largo_rect_eq else None
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
    det.metodo_largo = m.metodo
    if um_por_px and um_por_px > 0:
        det.area_um2 = m.area_um2
        det.largo_um = m.largo_um
        det.ancho_um = m.ancho_um
        det.feret_um = m.feret_um
        det.geodesico_um = m.geodesico_um
        det.largo_rect_eq_um = m.largo_rect_eq_um
        det.diam_um = 2.0 * float(np.sqrt(m.area_um2 / np.pi)) if m.area_um2 else None
    return True
