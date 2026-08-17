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

# Colores polímero (Nile Red bajo UV 254 nm)
CLASS_COLOR_HEX = {
    "PET": "#e3342f",    # Rojo
    "PP": "#ff8c00",     # Naranjo
    "LDPE": "#ffd700",   # Amarillo
}
```

**Fuente:** Segoe UI (Windows-native)

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

### 🏷 Etiquetador (Module 3, en construcción)
Anotación YOLO con pre-anotación automática.

### 📐 Visor (Module 4, en construcción)
Inspección interactiva de una imagen con calibración μm/píxel.

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
    diam_um: Optional[float] # Diámetro equivalente (μm) con calibración
    area_um2: Optional[float]
```

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
- 🏗 Etiquetador: En construcción
- 🏗 Visor: En construcción
- 📋 Validación de dataset (duplicados, outliers)
- 🤖 Recomendaciones automáticas de parámetros

---

## 📞 Contacto

**Cristofher Ferrada**  
Doctorado en Ciencias mención Química — Pontificia Universidad Católica de Valparaíso  
2026

**Repo:** https://github.com/CrissFerrada/Poly-X-Microplastics
