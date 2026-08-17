# 🚧 En Construcción — Sugerencias y Próximos Pasos

**Proyecto:** Poly-X 🔬 (Suite de detección de microplásticos)
**Estado actual:** v2.0.0 · YOLO v8/v11 + Nile Red · 2 papers publicados · repo en GitHub
**Meta:** ser la suite de referencia para detección automatizada de microplásticos por fluorescencia
**Última revisión:** 04-07-2026

---

## 🎯 Diagnóstico

Es tu proyecto científico más **consolidado**: ya respaldado por dos publicaciones (PLoS ONE 2024,
J. Chil. Chem. Soc. 2024) y con repositorio en GitHub. La oportunidad es convertirlo en referencia
comparativa (benchmark) para otros grupos de investigación.

---

## ✅ Próximos pasos (orden de prioridad)

### 1. Terminar los módulos "en construcción" (ALTA)
- [ ] Completar el **Etiquetador** (anotación YOLO con pre-anotación automática)
- [ ] Completar el **Visor** (inspección con calibración μm/píxel)
- **Por qué:** sin ellos, el flujo entrenamiento→etiquetado→detección queda a medias para un usuario externo.

### 2. Publicar el dataset con DOI (ALTA)
- [ ] Liberar el dataset YOLO etiquetado (PET/PP/LDPE) en Zenodo o Roboflow con DOI
- **Por qué:** un dataset de microplásticos por fluorescencia es escaso; citarlo da impacto y visibilidad.

### 3. Reportar métricas de referencia (benchmark) (MEDIA)
- [ ] Publicar en el README mAP, Precision y Recall por clase del modelo de producción
- [ ] Documentar hardware y parámetros usados para reproducibilidad
- **Por qué:** convierte tu suite en punto de comparación estándar para otros equipos.

### 4. Validación de dataset (MEDIA)
- [ ] Implementar detección de duplicados y outliers (ya listado como pendiente en el README)
- [ ] Cerrar los 20+ scripts `_diag_*.py` en una herramienta de auditoría unificada

---

## 🚫 No hacer todavía
- Nuevas clases de polímero o modelos exóticos antes de cerrar Etiquetador + Visor y publicar el dataset.
