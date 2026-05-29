# Poly-X — Suite de detección de microplásticos

> **Detección automatizada y clasificación de microplásticos (PET, PP, LDPE)  
> por fluorescencia Nile Red bajo luz UV (254 nm) con modelos YOLO v8/v11.**

**Autor:** Cristofher Ferrada · Doctorado en Ciencias mención Química · PUCV · 2026  
**Versión:** 2.0.0 · Windows 10/11 · Python 3.11

---

## Módulos

| # | Módulo | Estado | Descripción |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Funcional | Menú principal — lanza cada módulo en ventana independiente |
| 2 | 🔬 **Detector** | ✅ Funcional | Análisis en lote con YOLO · reporte HTML paper-quality |
| 3 | 🎯 **Entrenador** | ✅ Funcional | Entrenamiento YOLO v8/v11 con curvas en vivo |
| 4 | 🏷 **Etiquetador** | ✅ Funcional | Anotación YOLO interactiva con pre-anotación automática |
| 5 | 📐 **Visor** | ✅ Funcional | Inspección de imágenes con calibración interactiva μm/píxel |

---

## Características

### 🔬 Detector (análisis en lote)
- Carga hasta **3 modelos .pt** en paralelo para comparación directa
- Inferencia sobre carpetas de imágenes con barra de progreso
- Análisis de errores: **TP/FP/FN**, matriz de confusión, galería de falsos positivos/negativos
- Calibración **μm/px** para reportar tamaños reales de partículas
- **Histograma de distribución de tamaños** por clase (PET/PP/LDPE) — visible cuando hay calibración activa
- **Exportar CSV** directo desde la página de resultados
- **Drag & Drop** de modelos `.pt` e imágenes/carpetas
- Reporte HTML autocontenido (imágenes en base64, calidad paper)

### 🎯 Entrenador
- Soporte para **YOLO v8 y v11**, tamaños nano → xlarge
- Carga de `data.yaml` con auto-detección de splits train/val/test
- **Validación del dataset** con indicadores ✓/✗ (imágenes, labels, clases)
- Curvas de pérdida en vivo (Box loss, mAP, Precision, Recall)
- Augmentación configurable (flip, rotation, mosaic, HSV)
- Exportación a **ONNX / TensorRT / CoreML**
- Diagnóstico explícito de GPU/CPU al iniciar el entrenamiento

### 🏷 Etiquetador
- Canvas interactivo: **arrastrar** para dibujar cajas YOLO normalizadas
- Clic derecho para asignar clase; teclas `1`–`9` para cambiar clase activa
- **Undo/Redo** (Ctrl+Z / Ctrl+Y) por imagen
- **Pre-anotación automática** con modelo `.pt` (imagen actual o todas)
- **Auto-guardado silencioso** cada 60 segundos
- **Thumbnails** en la lista lateral de imágenes
- Guarda `.txt` YOLO junto a la imagen (o en `labels/` si está en `images/`)
- Genera `classes.txt` automáticamente para el Trainer

### 📐 Visor
- Abre una imagen individual o navega una carpeta con `← →`
- **Calibración interactiva μm/píxel** en 2 modos:
  - 📏 **Línea** (2 clics): marca una referencia conocida, ingresa su tamaño real
  - ⭕ **Círculo** (3 clics): marca 3 puntos del borde, ingresa el diámetro real
- Barra inferior muestra `📐 0.4880 μm/px (línea)` en tiempo real
- Detección con modelo `.pt` cargado (Ctrl+D)
- Tabla de resultados filtrable por clase con Ø(px) y Ø(μm)
- **Drag & Drop** de imagen o modelo `.pt` directo al canvas
- Exporta: imagen anotada en alta resolución + `detecciones.csv` + `resumen.json`

---

## Polímeros detectados

| Clase | Color (Nile Red, UV 254 nm) | ID |
|---|---|---|
| **PET** | 🔴 Rojo | 0 |
| **PP** | 🟠 Naranja | 1 |
| **LDPE** | 🟡 Amarillo | 2 |

---

## Requisitos

| Componente | Versión |
|---|---|
| Windows | 10 / 11 |
| Python | **3.11.x** (no 3.12+) |
| RAM | 8 GB mínimo |
| GPU | NVIDIA opcional (20–30× más rápido para entrenamiento) |

Dependencias principales: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`, `Matplotlib 3.9`

---

## Instalación

```bat
SETUP.bat
```

Crea el entorno `.venv`, detecta GPU NVIDIA y descarga PyTorch (CUDA o CPU automáticamente).

## Uso

```bat
iniciar_polyx.bat
```

Abre el **Launcher** → selecciona el módulo.

O directamente desde terminal:

```bash
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.detector
.venv\Scripts\python.exe -m polyx.trainer
.venv\Scripts\python.exe -m polyx.etiquetador
.venv\Scripts\python.exe -m polyx.visor
```

---

## Estructura del proyecto

```
polyx/
├── launcher.py          # Menú principal
├── core/                # Módulos compartidos (theme, yolo_wrap, metrics, report_html)
├── detector/            # Módulo 2: análisis en lote (9 páginas)
├── trainer/             # Módulo 3: entrenamiento YOLO (9 páginas)
├── etiquetador/         # Módulo 4: anotación interactiva
└── visor/               # Módulo 5: inspección + calibración μm/px
models/                  # Pesos .pt entrenados
runs/                    # Resultados de cada ejecución
data_microplastico/      # Dataset YOLO (images/ + labels/)
```

---

## Flujo completo de uso

```
Imágenes de microscopio (UV 254 nm, tinción Nile Red)
        ↓
  🏷 Etiquetador → anota PET/PP/LDPE en formato YOLO
        ↓
  🎯 Entrenador  → entrena modelo YOLO v8/v11
        ↓
  🔬 Detector    → analiza en lote + reporte HTML
        ↓
  📐 Visor       → inspección detallada + calibración μm/px
```

---

## Publicaciones

- **Pérez M, Parra S, Ferrada C, et al.** (2024). Detection and classification of microplastics using YOLO-based deep learning with Nile Red fluorescence staining. *PLoS ONE* **19**(12): e0314520. https://doi.org/10.1371/journal.pone.0314520

- **Ferrada C, Pérez M, Parra S, et al.** (2024). Automated identification of microplastic polymers by fluorescence and machine learning. *J. Chil. Chem. Soc.* **69**(1): 6082.

---

## Manual de usuario

El archivo `Manual_PolyX.html` contiene la documentación completa con capturas de cada módulo, atajos de teclado, flujos de trabajo recomendados y referencias bibliográficas.

---

## Licencia y contacto

**Cristofher Ferrada**  
Laboratorio de Química Ambiental · Pontificia Universidad Católica de Valparaíso  
[cristofher.ferrada@pucv.cl](mailto:cristofher.ferrada@pucv.cl)
