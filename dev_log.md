# Dev Log - Microplasticos

**Categoría**: Tesis  
**Última actualización**: 2026-06-07  
**Proyecto**: Poly-X — Suite de detección de microplásticos

## Descripción del Proyecto

**Poly-X** es una plataforma de detección automatizada y clasificación de microplásticos (PET, PP, LDPE) usando fluorescencia Nile Red bajo luz UV (254 nm) e IA con modelos YOLO v8/v11.

Integra flujo completo: entrenamiento → etiquetado → detección → reporte HTML.

**Publicaciones asociadas:**
- Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024). Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. PLoS ONE 19(12): e0314520
- Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024). Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in Merluccius gayi fish by Nile Red fluorescent staining and image analysis. J. Chil. Chem. Soc. 69(1): 6082-6085. https://doi.org/10.4067/s0717-97072024000106082

## Entorno de Trabajo

- **IDE/Editor**: PySide6 (GUI nativa Qt) + VS Code para scripts
- **Lenguajes**: Python 3.11.x
- **SO**: Windows 10/11 (compatibilidad macOS/Linux en desarrollo)
- **Hardware**: 8GB RAM mínimo, GPU NVIDIA recomendada (auto-detección CUDA)

### Dependencias clave

| Librería | Versión | Propósito |
|---|---|---|
| Ultralytics | 8.3.40 | YOLO inference + training |
| OpenCV | 4.10.0.84 | Procesamiento/anotación imágenes |
| PySide6 | 6.7.2 | GUI |
| NumPy | 1.26.4 | Cálculos numéricos |
| Matplotlib | 3.9.2 | Gráficas en reportes |
| pyqtgraph | 0.13.7 | Plots en vivo (trainer) |
| Pillow | 10.4.0 | Procesamiento PNG/JPEG |

### Notas especiales

- **Modelos preentrenados**: `bestdetectormedium.pt` (producción)
- **Colores NADES**: RGB de fluorescencia Nile Red (PET rojo, PP naranja, LDPE amarillo)
- **Rutas clave**: `models/`, `runs/`, `data_microplastico/`, `polyx/`

## Instrucciones para Codex

1. **Consultar CLAUDE.md** para arquitectura completa y flujo de datos
2. **Convenciones**:
   - Comentarios/variables: **Español**
   - Docstrings: Inglés (excepto clases muy específicas)
   - Rutas: siempre `pathlib.Path`, nunca strings
   - Colores: usar `theme.py` constants, nunca hardcodes
   - Qt Signals: en formato `snake_case` (ej: `models_changed`)

3. **Estructura principal** (en `polyx/`):
   - `launcher.py` — Menú principal (hero + 4 cards)
   - `core/` — Núcleo compartido (YOLO wrapper, métricas, tema)
   - `detector/` — Análisis en lote (8 pestañas: modelos, imágenes, parámetros, ejecución, resultados, errores, comparar, reporte)
   - `trainer/` — Entrenamiento YOLO (9 pestañas: modelo, dataset, parámetros, augmentación, entrenar, evaluar, comparar, exportar, informe)

4. **Para debugging**:
   - Ejecutar: `.venv\Scripts\python.exe -m polyx.launcher`
   - Logs en consola (Qt + Python logging)
   - Verificar CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## Estado Actual

### ✅ Completado
- Detector: Funcional (batch inference, comparación múltiples modelos, matriz confusión, HTML reporte)
- Entrenador: Funcional (curvas loss en vivo, métricas mAP/P/R/F1, exportación PT→ONNX)
- Infraestructura YOLO wrapper + métricas + reportes

### 🏗 En construcción
- Etiquetador (Module 3): GUI para anotación con pre-anotación automática
- Visor interactivo (Module 4): Inspección imagen + calibración μm/píxel

### 📋 Por hacer
- Validación automática de dataset (duplicados, outliers)
- Recomendaciones automáticas de parámetros según hardware
- Documentación video-tutorial

### ⚙️ Última acción
Restructuración arquitectura (mayo 2026): migración legacy → modular (launcher + core + detector + trainer).

---

*Este archivo se usa para sincronizar el contexto entre sesiones con Codex*
