# Poly-X — Suite de detección de microplásticos

**Estado:** Reconstrucción en curso (mayo 2026)  
**Autor:** Cristofher Ferrada · Doctorado en Ciencias mención Química · PUCV  
**Versión:** 2.0.0  
**GitHub:** https://github.com/CrissFerrada/Poly-X-Microplastics

## 🎯 Propósito

Plataforma de **detección automatizada y clasificación** de microplásticos (PET, PP, LDPE) por fluorescencia Nile Red bajo luz UV (254 nm) usando modelos YOLO v8/v11 (deep learning).

Integra todo el flujo: **entrenamiento → etiquetado → detección → reporte HTML paper-quality**.

### Publicaciones

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. PLoS ONE 19(12): e0314520. https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).** Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in Merluccius gayi fish by Nile Red fluorescent staining and image analysis. J. Chil. Chem. Soc. 69(1): 6082-6085. https://doi.org/10.4067/s0717-97072024000106082

---

## 📦 Estructura

```
polyx/                    # Código fuente
├── __init__.py           # v2.0.0
├── launcher.py           # Módulo 1: Menú principal (hero + 4 cards)
├── core/                 # Núcleo compartido
│   ├── theme.py          # Paleta de colores (Segoe UI, Nile Red RGB)
│   ├── paths.py          # Rutas (models/, runs/, data/, assets/)
│   ├── yolo_wrap.py      # Wrapper YOLO (Detection, YoloModel)
│   ├── metrics.py        # TP/FP/FN, IoU, matriz de confusión
│   ├── widgets.py        # LogoBadge, HLine, custom widgets
│   └── report_html.py    # Generador reporte (base64, paper-quality)
├── detector/             # Módulo 2: Análisis en lote
│   ├── __main__.py
│   ├── window.py         # Ventana principal (sidebar + 9 páginas)
│   ├── state.py          # DetectorState (modelos, imágenes, params)
│   ├── runner.py         # Lógica de ejecución (batch inference)
│   └── pages/
│       ├── _base.py      # BasePage (herencia)
│       ├── modelos.py    # Carga hasta 3 modelos .pt
│       ├── imagenes.py   # Selecciona carpeta de imágenes
│       ├── parametros.py # Conf, IoU, um/px, device (CPU/GPU)
│       ├── gt_manual.py  # Anota ground truth manual
│       ├── ejecutar.py   # Botón ▶ run + progreso
│       ├── resultados.py # Tabla: imagen, clase, conf, tamaño
│       ├── errores.py    # Matriz confusión, galería FP/FN
│       ├── comparar.py   # Compara runs (curvas P/R/F1)
│       └── reporte.py    # Genera HTML + abre navegador
├── trainer/              # Módulo 3: Entrenamiento YOLO
│   ├── __main__.py
│   ├── window.py         # Ventana principal (sidebar + 9 páginas)
│   ├── state.py          # TrainerState (config, runs, checkpoints)
│   ├── runner.py         # Lógica de entrenamiento
│   ├── hw.py             # Detección GPU, hardware info
│   └── pages/
│       ├── _base.py
│       ├── modelo.py     # YOLO v8/v11, tamaño (nano-xlarge)
│       ├── dataset.py    # Dataset YAML + split (train/val/test)
│       ├── parametros.py # Epochs, batch, imgsz, optimizer
│       ├── augmentacion.py # Flip, rotation, mosaic, etc.
│       ├── entrenar.py   # Lanzar training + loss curves en vivo
│       ├── evaluar.py    # Métricas mAP, P, R, F1 por clase
│       ├── comparar.py   # Comparar múltiples runs
│       ├── exportar.py   # Export PT→ONNX/TensorRT/CoreML
│       └── informe.py    # Reporte entrenamiento
└── legacy/               # Versiones anteriores (referencia)
    ├── Detector_Microplastico.py
    ├── trainer_microplastico.py
    ├── polyx_viewer.py
    └── ...

models/                   # Pesos YOLO .pt entrenados
├── bestdetectormedium.pt # Modelo por defecto (producción)
└── ...

runs/                     # Resultados de cada ejecución
└── YYYYMMDD_HHMMSS/
    ├── images/          # Imágenes anotadas (PNG)
    ├── centroids.csv    # class, x, y, conf, diam_um
    ├── metrics.json     # TP/FP/FN por clase
    └── annotations/     # YOLO .txt si hay GT

data_microplastico/       # Dataset etiquetado YOLO
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/           # .txt YOLO (class cx cy w h norm)
│   └── val/
└── classes.txt          # 0: PET, 1: PP, 2: LDPE

data/                     # Otros datasets
```

---

## 🔧 Tecnología

| Componente | Versión | Propósito |
|---|---|---|
| **Python** | 3.11.x | Runtime |
| **PySide6** | 6.7.2 | GUI (Qt) |
| **Ultralytics** | 8.3.40 | YOLO inference + training |
| **OpenCV** | 4.10.0.84 | Lectura/anotación imágenes |
| **NumPy** | 1.26.4 | Cálculos |
| **Pillow** | 10.4.0 | Procesamiento PNG/JPEG |
| **Matplotlib** | 3.9.2 | Gráficas (reportes) |
| **pyqtgraph** | 0.13.7 | Plots en vivo (trainer) |

**Hardware:**
- Windows 10/11
- 8 GB RAM mínimo
- GPU NVIDIA opcional (recomendada): Auto-detecta CUDA → PyTorch GPU

---

## 🎨 Paleta (theme.py)

Coincide con Manual_PolyX.html:

```python
INK = "#1f2328"          # Texto principal
INK2 = "#424a53"         # Subtítulos
INK3 = "#656d76"         # Muted
ACCENT = "#0969da"       # Botones primarios (azul)
OK = "#1f6b5e"           # Verde (positivo)
WARN = "#9a6700"         # Naranjo (warning)
ERR = "#cf222e"          # Rojo (error)
VIO = "#6639ba"          # Púrpura

# Colores con que se dibujan las cajas de cada clase en pantalla.
# OJO: son colores de interfaz, NO describen la emisión real del Nile Red.
CLASS_COLOR_HEX = {
    "PET": "#e3342f",    # se dibuja rojo
    "PP": "#ff8c00",     # se dibuja naranjo
    "LDPE": "#ffd700",   # se dibuja amarillo
}
```

**Fuente:** Segoe UI (Windows-native)

### Fluorescencia real de cada polímero

No confundir con los colores de interfaz de arriba. Medido sobre el interior de
las cajas del dataset de entrenamiento (media RGB, n=30 por clase):

| Clase | R | G | B | Aspecto |
|---|---|---|---|---|
| PET | 116 | 58 | 65 | rojo–salmón |
| PP | 122 | **125** | 32 | **amarillo verdoso, apagado** |
| LDPE | 181 | 162 | 57 | amarillo franco, brillante |

**PP y LDPE comparten tono y se separan por brillo** (R 122 frente a 181), más el
matiz verdoso del PP. Es la confusión dominante al anotar y explica que el recall
por clase se hunda ahí: PET 0.98 · PP 0.70 · LDPE 0.54.

La documentación decía que PP era «naranjo», lo que no corresponde a la emisión
observada. Corregido en agosto de 2026 a partir de la medición y de la
observación del autor.

---

## 🚀 Instalación y uso

### Primera vez
```batch
SETUP.bat
```
Crea `.venv`, detecta GPU, instala PyTorch + ultralytics + PySide6.

### Uso diario
```batch
iniciar_polyx.bat
```
Abre **Launcher** → selecciona módulo.

---

## 📊 Módulos principales

### 🔬 Detector (Module 1)
**Propósito:** Análisis en lote de imágenes con modelo entrenado.

**Flujo típico:**
1. **Modelos** → Cargar hasta 3 modelos `.pt` (compara en paralelo)
2. **Imágenes** → Seleccionar carpeta (recursivo)
3. **GT manual** (opcional) → Anotar ground truth manualmente
4. **Parámetros** → conf=0.25, IoU=0.45, um/px=2.5 (calibración)
5. **Ejecutar** → Batch inference con progreso
6. **Resultados** → Tabla (clase, conf, tamaño)
7. **Errores** → Matriz confusión + galería FP/FN
8. **Comparar** → Compara runs (curvas P/R/F1)
9. **Reporte** → HTML self-contained + navegador

**Output:**
- `runs/YYYYMMDD_HHMMSS/images/*.png` — Imágenes anotadas
- `runs/.../centroids.csv` — class, x, y, conf, diam_um
- `runs/.../metrics.json` — TP/FP/FN por clase
- `runs/.../report.html` — Reporte paper-quality (base64)

### 🎯 Entrenador (Module 2)
**Propósito:** Entrenar modelos YOLO v8/v11 desde cero o fine-tuning.

**Flujo:**
1. **Modelo** → YOLO v8/v11, tamaño (nano, small, medium, large, xlarge)
2. **Dataset** → Dataset.yaml (train/val/test split)
3. **Parámetros** → Epochs, batch, imgsz, optimizer (SGD, Adam)
4. **Augmentación** → Flip, rotation, mosaic, HSV, etc.
5. **Entrenar** → Lanzar entrenamiento + loss curves en vivo
6. **Evaluar** → mAP, P, R, F1 por clase (validación)
7. **Comparar** → Compara múltiples runs
8. **Exportar** → PT → ONNX/TensorRT/CoreML
9. **Informe** → Reporte de entrenamiento

**Output:**
- `runs/detect/trainN/weights/best.pt` — Modelo óptimo
- `runs/detect/trainN/results.csv` — Métricas por época
- `runs/detect/trainN/confusion_matrix.png` — Matriz confusión

### 🏷 Etiquetador (Module 3, funcional)
Anotación YOLO manual con pre-anotación automática opcional.

**Verificado con 552 recortes** (agosto 2026). Arranque 0.25 s, interfaz
responde en 0.06 s.

**Seguimiento del avance.** La lista distingue tres estados, y se recupera del
disco al abrir la carpeta, así que el conteo puede repartirse en varias sesiones:

| Marca | Significado |
|---|---|
| `✓ nombre (n)` | revisada, con n partículas |
| `· nombre (0)` | revisada, sin partículas — **es un dato** |
| `○ nombre` | todavía sin revisar |

Un `.txt` **solo** se escribe al marcar la imagen como revisada o al dibujar una
caja. Pasar de largo no crea archivo: registrar como "revisada con cero" una
imagen apenas ojeada falsearía el censo.

**Atajos:** `Espacio` marca revisada y avanza · `Tab` salta a la siguiente sin
revisar · `F` reencuadra · `1-9` clase activa · `←/→` navegar · `Supr` borra la
caja seleccionada · `Ctrl+Z/Y` deshacer/rehacer · rueda zoom · botón medio pan.

**Lado mínimo de caja: 2 px** (`BboxCanvas.LADO_MINIMO_PX`). Estaba en 5 px y
descartaba **en silencio** marcas legítimas: las partículas más pequeñas del
estudio miden ~8 px de lado. Ahora, si una caja se rechaza, se avisa en la barra
de estado.

El zoom se conserva entre imágenes (casilla en el panel derecho), con tolerancia
de tamaño: los recortes de una rejilla difieren en 1 px por redondeo y exigir
igualdad exacta hacía perder el zoom en cada cambio.

### 📐 Visor (Module 4, funcional)
Inspección de una imagen con calibración μm/píxel, detección y exportación.

**Calibración interactiva.** Dos modos: **línea** (2 clics + longitud real) y
**círculo** (3 clics sobre un borde circular + diámetro real). El diálogo del
círculo viene con 100000 μm por defecto, que es la placa Petri del estudio.

> Estuvo inutilizable hasta agosto 2026: `QInputDialog.getDouble()` se llamaba
> con `min=`/`max=`, que PySide6 no acepta como palabras clave, y lanzaba
> `AttributeError` al completar los puntos. La matemática siempre estuvo bien;
> el diálogo reventaba antes de llegar a ella. **Los argumentos deben ir
> posicionales:** `(parent, title, label, value, minValue, maxValue, decimals)`.

**Detección.** `imgsz` configurable (320–8192, por defecto 2080) y GPU si está
disponible. Antes forzaba `device="cpu"` e `imgsz` por defecto 640: con
partículas de ~12 px en fotos de 4096 px, a 640 colapsan a ~2 px y **no se
detecta nada**. Si la GPU se queda sin memoria, se explica cómo bajar `imgsz` en
vez de mostrar la excepción cruda.

**Cargar etiquetas (.txt).** Muestra las anotaciones YOLO que acompañan a la
imagen, con las tallas ya convertidas a μm. Sirve para revisar el conteo manual
sobre la placa completa sin volver al Etiquetador.

**Exportación** a `visor_<imagen>_<ts>/`: PNG anotado, `detecciones.csv`
(una fila por partícula, con diam_px, diam_um y area_um2) y `resumen.json`.

> Lectura y escritura de imágenes con `cv2.imdecode`/`imencode` sobre
> `np.fromfile`, no `imread`/`imwrite`: en Windows estos fallan con rutas que
> llevan acentos y devuelven `None` sin avisar.

---

## 🔑 Clases clave

### `Detection` (yolo_wrap.py)
```python
@dataclass
class Detection:
    class_id: int           # 0=PET, 1=PP, 2=LDPE
    class_name: str         # "PET", "PP", "LDPE"
    conf: float             # 0-1, confianza
    x1, y1, x2, y2: float   # Bbox píxeles (absolutos)
    diam_um: Optional[float]  # Diámetro del círculo de igual área REAL
    area_um2: Optional[float] # Área de la partícula, no de la caja
    largo_um: Optional[float] # Dimensión mayor, siguiendo la curva
    ancho_um: Optional[float]
    aspecto: Optional[float]    # largo/ancho reales
    curvatura: Optional[float]  # 1.0 = recta; ≥1.15 = curva
    morfotipo: Optional[str]    # "fibra" | "fragmento"
```

**La caja no describe la forma.** Medido sobre 7.129 partículas anotadas: la caja
sobreestima el área **1.87×**, y una fibra tumbada en diagonal tiene caja
**cuadrada** — con lo que se reportaba como fragmento.

`morfologia.py` segmenta dentro de cada caja (Otsu sobre el canal de mayor
contraste, componente conexa del centro, recortada a la caja) y mide sobre la
máscara. **El largo sale de dos medidas estándar y se reporta la mayor:**

| Medida | Qué es | Falla en |
|---|---|---|
| **Feret máximo** | mayor distancia entre dos puntos del borde | fibras curvas (da la cuerda: −34.6 % en un arco de 180°) |
| **Diámetro geodésico** | camino más largo *dentro* de la partícula | partículas gruesas (rodea las concavidades) |

El geodésico solo se usa si la partícula cumple **dos** condiciones, y cada una
viene de un fallo observado:

| Condición | Constante | Por qué |
|---|---|---|
| **Delgada**, `largo/grosor ≥ 4` | `DELGADEZ_PARA_GEODESICO` | en una gruesa el camino rodea las concavidades: un grumo real de 44 px recibía 73 |
| **No convexa**, `solidez < 0.90` | `SOLIDEZ_CONVEXA` | en un convexo geodésico y Feret coinciden por definición; una barra girada medía 200 px a 0° y 45° pero **208 a 15°, 30° y 60°** |

Contra formas sintéticas de talla conocida: **error mediano 0.6 %, peor caso
4.7 %**. Fijado en `tests/test_morfologia.py`.

**El rectángulo equivalente `L = (P + √(P²−16A))/4` NO es una talla** y no debe
usarse como tal. Depende del perímetro, así que un borde dentado lo infla
(+22.5 % en una recta con dientes de sierra), y solo está definido si P² ≥ 16A
—un círculo da P² = 4πA y el discriminante sale negativo—. Se conserva en
`largo_rect_eq` como descriptor: comparado con los otros dos delata bordes
irregulares.

> **Corrección de agosto 2026.** Se reportó que el material tenía «22 % de fibras
> y aspecto hasta 21.1». Era un artefacto de usar `rect_eq` como largo. Medido
> con Feret/geodésico sobre 6.638 partículas: **aspecto mediano 1.58, máximo 8.7,
> y 1.1 % de fibras**. El material es mayoritariamente fragmentos compactos.

Coste 5.9 ms por partícula. Solo cv2 y numpy: `scipy` **no** está en
`requirements.txt` —llega de rebote con ultralytics— y apoyarse en él rompería
la instalación el día que ultralytics deje de traerlo.

### `YoloModel` (yolo_wrap.py)
Wrapper lazy-loading de Ultralytics YOLO:
```python
model = YoloModel("bestdetectormedium.pt")
detections = model.predict("image.jpg", conf=0.25, device="0")
```

### `DetectorState` (detector/state.py)
Estado global del Detector (Qt signals):
```python
state.model_slots: List[ModelSlot]      # Hasta 3 modelos
state.images: List[Path]                # Imágenes a analizar
state.params: InferenceParams           # conf, IoU, um_per_px
state.results: Dict[int, List[ImageResult]]  # Resultados por modelo
```

### `ImageResult` (detector/state.py)
Resultado de una imagen:
```python
image_path: Path
predictions: List[Detection]
gt: List[Detection]             # Ground truth (si existe)
tp, fp, fn, miscls: int
annotated_png: Optional[bytes]  # PNG anotado en base64
```

### `MatchResult` (metrics.py)
Matching pred vs GT (IoU ≥ 0.5):
```python
tp: int          # True Positives (conf y clase correcta)
fp: int          # False Positives (sin GT)
fn: int          # False Negatives (GT sin pred)
miscls: int      # Bien localizado pero clase incorrecta
```

---

## 🔄 Flujo de datos

### Detector
```
Imágenes (carpeta)
    ↓
YoloModel.predict() × N modelos
    ↓
Detection[] (bbox + clase + conf)
    ↓
read_yolo_txt() → Ground truth (si existe)
    ↓
match_image() → TP/FP/FN (IoU ≥ 0.5)
    ↓
compute_box_size_um() → diam_um (si um_per_px calibrado)
    ↓
ImageResult
    ↓
cv2.rectangle() + anotación → PNG
    ↓
runs/YYYYMMDD_HHMMSS/
```

### Trainer
```
Dataset (images/ + labels/ YOLO)
    ↓
YOLO(model_name).train(epochs=..., batch=...)
    ↓
Ultralytics training loop
    ↓
best.pt (checkpoint óptimo)
    ↓
results.csv + curves
    ↓
runs/detect/trainN/
```

---

## 📝 Convenciones

- **Español:** Todos los comentarios, variables locales, UI en español
- **Inglés:** Docstrings en inglés (excepto clase/método si es muy específico)
- **Qt Signals:** `models_changed`, `run_started`, `run_finished`, etc.
- **Rutas:** Siempre `Path` de pathlib, nunca strings
- **Colores:** Usar `T.ACCENT`, `T.OK`, etc., nunca hardcodes
- **Métodos:** `_` privado (widget, signal callback)

---

## 🔗 Dependencias clave

- **Ultralytics YOLO** → Inference, training, export
- **OpenCV** → `cv2.rectangle()`, anotación, lectura imágenes
- **NumPy** → Cálculos de bbox, IoU
- **Matplotlib** → Gráficas en reportes (en modo Agg, no interactivo)
- **PySide6/Qt** → GUI (signals/slots, QObject)

---

## ⚙️ Parámetros comunes

### Inferencia (DetectorState.params)
```python
conf: float = 0.25          # Umbral confianza
iou_nms: float = 0.45       # IoU para NMS
iou_tp: float = 0.50        # IoU para análisis de errores
imgsz: int = 640            # Tamaño entrada red
device: str = "0"           # "0" (GPU), "cpu"
um_per_px: float = 0.0      # Calibración (0 = sin calibrar)
size_min_um: float = 0.0    # Filtro inferior (0 = sin filtro)
size_max_um: float = 0.0    # Filtro superior
```

### Entrenamiento
```python
epochs: int = 100
batch: int = 16             # Menor si OOM
imgsz: int = 640            # Menor si OOM
device: str = "0"
optimizer: str = "SGD"      # o "Adam"
lr0: float = 0.01
```

---

## 🐛 Troubleshooting

| Problema | Solución |
|---|---|
| "No module named tkinter" | Reinstalar Python, marcar tcl/tk |
| "CUDA not available" | Normal, solo CPU funciona (más lento) |
| Detector no detecta nada | Bajar conf a 0.10 en Parámetros |
| OOM en entrenamiento | Bajar batch a 4-8, imgsz a 512-576 |
| Reporte no abre | Revisar Manual_PolyX.html manualmente |

---

## 📚 Archivos generados automáticamente

- **Manual_PolyX.html** — Generado por `generar_manual.py` (screenshots + especificación)
- **manual_screenshots/** — Capturas de cada tab (actualizadas al regenerar manual)

Para regenerar:
```bash
.venv\Scripts\python.exe generar_manual.py --solo detector
.venv\Scripts\python.exe generar_manual.py              # todos
```

---

## 🔮 Próximas mejoras (en construcción)

- ✅ Detector: Funcional, reporte HTML
- ✅ Entrenador: Funcional, curvas en vivo
- ✅ Etiquetador: Funcional, verificado con 552 recortes (ago 2026)
- ✅ Visor: Funcional, calibración reparada y verificada (ago 2026)
- 📋 Validación de dataset (duplicados, outliers)
- 🤖 Recomendaciones automáticas de parámetros

---

## 📞 Contacto

**Cristofher Ferrada**  
Doctorado en Ciencias mención Química — Pontificia Universidad Católica de Valparaíso  
2026

**Repo:** https://github.com/CrissFerrada/Poly-X-Microplastics
