# Poly-X — Suite de detección de microplásticos

***Español** · [English](README.en.md)*

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
- Análisis de errores: **Verdaderos Positivos / Falsos Positivos / Falsos Negativos / Mal Clasificados**, matriz de confusión, galería de falsos positivos/negativos
- Calibración **μm/px** para reportar tamaños reales de partículas
- **Histograma de distribución de tamaños** por clase (PET/PP/LDPE) — visible cuando hay calibración activa
- **Exportar CSV** directo desde la página de resultados
- **Drag & Drop** de modelos `.pt` e imágenes/carpetas
- **Resolución de inferencia** ajustable con perfiles (Rápido 1280 · Equilibrado
  2560 · Máxima detección 4096) y un botón que mide el máximo que aguanta tu GPU
- Reporte HTML autocontenido (imágenes incrustadas en **base64**, calidad paper)
  con **galería comparativa Predicción vs Ground Truth (lado a lado)**. Las
  imágenes de galería se recodifican y se limita su número para que el archivo
  siga siendo abrible en un navegador; las métricas cubren todas las imágenes
- **Exportación a PDF** del reporte (un clic), listo para enviar por correo

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
- **Seguimiento del avance**, pensado para campañas de cientos de imágenes
  repartidas en varias sesiones. El estado se recupera del disco al abrir la
  carpeta y la lista distingue tres casos:

  | Marca | Significado |
  |---|---|
  | `✓ nombre (n)` | revisada, con n objetos |
  | `· nombre (0)` | revisada, sin objetos — **es un dato** |
  | `○ nombre` | todavía sin revisar |

  Un `.txt` solo se escribe al marcar la imagen como revisada o al dibujar una
  caja: pasar de largo no crea archivo, porque registrar como «revisada con
  cero» una imagen apenas ojeada falsearía un conteo censal.
- `Espacio` marca revisada y avanza · `Tab` salta a la siguiente sin revisar ·
  `F` reencuadra · `←/→` navega · `Supr` borra la caja seleccionada
- **El zoom se conserva** entre imágenes (casilla en el panel derecho)
- **Lado mínimo de caja: 2 px.** Al rechazar una caja se avisa en la barra de
  estado, en vez de descartarla en silencio
- **Pre-anotación automática** con modelo `.pt`, con `conf` e `imgsz` ajustables
  y GPU si está disponible
- **Auto-guardado silencioso** cada 60 segundos; miniaturas generadas en segundo
  plano para que la ventana responda de inmediato
- Guarda `.txt` YOLO junto a la imagen (o en `labels/` si está en `images/`) y
  genera `classes.txt` en la raíz y en cada subcarpeta

### 📐 Visor
- Abre una imagen individual o navega una carpeta con `← →`
- **Calibración interactiva μm/píxel** en 2 modos:
  - 📏 **Línea** (2 clics): marca una referencia conocida, ingresa su tamaño real
  - ⭕ **Círculo** (3 clics): marca 3 puntos del borde, ingresa el diámetro real
    (útil con placas Petri, cuyo diámetro es conocido)
- Barra inferior muestra `📐 0.4880 μm/px (línea)` en tiempo real
- Detección con modelo `.pt` cargado, con **resolución de inferencia
  configurable** (320–8192) y GPU si está disponible. Con objetos diminutos en
  fotos de alta resolución el `imgsz` es determinante: a valores bajos las
  partículas caen por debajo del *stride* de la red y no se detecta nada
- **Cargar etiquetas `.txt`**: muestra anotaciones ya existentes sobre la imagen,
  con sus tallas convertidas a μm. Sirve para revisar un conteo manual sin
  volver al Etiquetador
- Tabla de resultados filtrable por clase con Ø(px) y Ø(μm)
- **Drag & Drop** de imagen o modelo `.pt` directo al canvas
- Exporta: imagen anotada + `detecciones.csv` + `resumen.json`

---

## Polímeros detectados

| ID | Clase | Fluorescencia observada (Nile Red, UV) | Color en la interfaz |
|---|---|---|---|
| 0 | **PET** | Rojo–salmón | 🔴 `#e3342f` |
| 1 | **PP** | **Amarillo verdoso, apagado** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Amarillo franco y **más brillante** | 🟡 `#ffd700` |

Los colores de la derecha son solo los de las cajas en pantalla; no describen la
emisión real. Medido sobre el interior de las cajas del dataset de entrenamiento
(media RGB, n=30 por clase):

| Clase | R | G | B |
|---|---|---|---|
| PET | 116 | 58 | 65 |
| PP | 122 | **125** | 32 |
| LDPE | 181 | 162 | 57 |

**PP y LDPE no se distinguen por tono sino por brillo**: ambos son amarillentos,
pero en PP el verde iguala o supera al rojo y la emisión es bastante más apagada.
Es la confusión más habitual al anotar, y la razón de que el recall por clase caiga
en esas dos.

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

Tras descargar el proyecto desde GitHub (botón **Code → Download ZIP**, y descomprimir;
o `git clone`), haz **doble clic en**:

```bat
SETUP.bat
```

El instalador:

1. **Pregunta dónde instalar Poly-X** (pulsa ENTER para instalarlo en la misma carpeta, o escribe otra ruta).
2. Detecta Python 3.11 y crea el entorno `.venv`.
3. Detecta GPU NVIDIA y descarga PyTorch (CUDA o CPU automáticamente).
4. Instala el resto de dependencias y verifica que todo importa.
5. Te ofrece **crear un acceso directo "Poly-X" en el Escritorio** y **te dice cómo iniciarlo**.

### Abrir sin consola

Doble clic en **`Poly-X.vbs`** (raíz del proyecto): abre el Launcher directamente,
sin ventana negra. Resuelve su propia ubicación, así que funciona desde cualquier
carpeta — basta descargar el repositorio y hacer doble clic, sin editar rutas.

Para tenerlo a mano, clic derecho sobre `Poly-X.vbs` → *Enviar a* → *Escritorio
(crear acceso directo)*, y asígnale el icono `assets\polyx.ico`.

`iniciar_polyx.bat` hace lo mismo pero mostrando la consola, útil para ver
mensajes de error si algo falla al arrancar.

> **Importante:** los modelos entrenados (`*.pt`) **no** se incluyen en la descarga de GitHub
> por su tamaño. Copia tu archivo `.pt` dentro de la carpeta `models\` para usar el Detector y el Visor.

> Requiere **Python 3.11.x** (no 3.12+). Si no lo tienes, descárgalo desde
> [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
> marcando *"Add Python to PATH"*.

## Uso

Doble clic en el acceso directo **Poly-X** del Escritorio, o en:

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

## Actualizar

Para traer la última versión publicada en GitHub **sin reinstalar nada**, doble clic en:

```bat
actualizar.bat
```

Comprueba si hay un commit nuevo en `main`; si lo hay, descarga y reemplaza solo los
archivos del programa. **Conserva** tu entorno `.venv`, tus modelos `models\*.pt`, tus
`runs\` y cualquier dato local. No necesita tener Git instalado (descarga por HTTPS) —
solo conexión a internet.

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
