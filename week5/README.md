# Semana 5 — Entrega (Deep Learning)

**Curso:** Deep Learning - Conceptos (601539) · **Programa:** FU · CAD2202023205 · EIAIPA2026  

**Producto:** notebook ejecutable en **Google Colab**, con evidencia versionada en **GitHub** (esta carpeta).

---

## Objetivo (consigna)

Profundizar en **hiperparámetros** y **optimización** del entrenamiento de una red neuronal, **evidenciando** cómo cambia el comportamiento al ajustar (por ejemplo) **tasa de aprendizaje**, **tamaño de lote** y **número de épocas**, y/o al cambiar el **optimizador**, manteniendo **fijos** el resto de elementos en cada comparación para que sea **válida**. La evidencia son resultados **verificables** (tablas, gráficos, impresiones por época) y **conclusiones breves** en Markdown.

---

## Archivos en esta carpeta

| Archivo | Rol |
|---------|-----|
| **[`actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb`](actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb)** | **Entrega principal:** modelo entrenable, entrenamiento por épocas, comparaciones y análisis |
| [`actividad5/README.md`](actividad5/README.md) | Detalle técnico (secciones del notebook, tabla de experimentos) |

**Cumplimiento mínimo (rúbrica):** el notebook incluye **más de dos** configuraciones comparadas (optimizadores; barrido de `learning_rate`; barrido de `batch_size`), con **loss** y **accuracy** por época (train/val), métricas en test y una sección final de conclusiones sobre **estabilidad**, **velocidad de convergencia** y **hallazgos**.

---

## Qué se comparó (resumen)

1. **Optimización:** `SGD` (con momentum) vs `Adam` vs `RMSprop` — mismas épocas (80), mismo `batch_size` (32), misma arquitectura y datos; solo cambia la regla de optimización (y el `learning_rate` documentado por optimizador, práctica habitual).
2. **Hiperparámetro `learning_rate`:** tres valores con **Adam** fijo — resto igual.
3. **Hiperparámetro `batch_size`:** tres valores con **Adam** y `lr` fijos — resto igual.

*(Así se cumple “al menos dos configuraciones” con controles explícitos.)*

---

## Resultado principal (qué esperar)

Los valores exactos dependen de la corrida; en el notebook se obtiene una **tabla** de `test_accuracy` / `test_loss` y **gráficos** de `loss` y `accuracy` vs época por grupo de experimentos. Suele observarse que **Adam/RMSprop** convergen con distinta suavidad que **SGD**, que un **`lr` demasiado alto** puede inestabilizar el entrenamiento y uno **muy bajo** ralentiza la mejora en el mismo número de épocas, y que el **batch** modifica el ruido del gradiente y el número de actualizaciones por época. **Redacte** las conclusiones concretas en la **sección 7** del notebook tras ejecutar.

---

## Cómo ejecutar el notebook

1. Subir o abrir [`actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb`](actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb) en [Google Colab](https://colab.research.google.com/) (desde GitHub: *Open in Colab* si lo configura, o descargar/subir el archivo).
2. **Entorno → Ejecutar todo** (o *Runtime → Run all*).
3. La primera celda instala dependencias (`tensorflow`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`) si hace falta.
4. Revisar salidas: impresiones por bloque de experimentos, **tres figuras** (optimizadores, LR, batch) y la **tabla** resumen.

**Local (Windows):** Si `pip install tensorflow` falla con **OSError** / rutas largas, la instalación puede quedar rota (`No module named 'tensorflow.python'`). Soluciones: activar [soporte de rutas largas](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation), usar un **venv** en una ruta corta (p. ej. `C:\venvs\dl`), o ejecutar en **Colab**.

**Local:** `pip install tensorflow scikit-learn matplotlib numpy pandas` (tras arreglar el tema de rutas si aplica).

---

## Enlaces para entregar y presentar

| Uso | Enlace |
|-----|--------|
| **Carpeta `week5/` en GitHub** (consigna) | [https://github.com/SonicWD/deep_learning/tree/main/week5](https://github.com/SonicWD/deep_learning/tree/main/week5) |
| **Notebook en Google Colab** (ejecutar sin instalar TF en tu PC) | [Abrir en Colab](https://colab.research.google.com/github/SonicWD/deep_learning/blob/main/week5/actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb) |

*(Los enlaces funcionan cuando `week5/` esté subido a la rama `main` del repo [SonicWD/deep_learning](https://github.com/SonicWD/deep_learning). Si usas otra rama o fork, sustituye en la URL.)*

### Checklist antes de entregar

1. `git add week5/` (o la ruta equivalente en tu clon), `commit`, `push` al repo público.
2. Abrir el enlace de la **carpeta** y comprobar que se ve este `README.md` y `actividad5/*.ipynb`.
3. En **Colab**: *Entorno → Ejecutar todo*; rellenar la **sección 7** del notebook con tus conclusiones (estabilidad, convergencia, hallazgos).
4. Opcional: volver a guardar el `.ipynb` **con las salidas** generadas en Colab y hacer un segundo `push` para dejar la evidencia en GitHub.

---

*Especialización en Inteligencia Artificial · Wilson Alfonso Diaz Capador*
