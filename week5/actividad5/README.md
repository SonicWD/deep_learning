# Semana 5 – Actividad 5: Clasificación multiclase con MLP (Keras, sin convolución)

**Curso:** Deep Learning - Conceptos (601539)  
**Programa:** FU · CAD2202023205 · EIAIPA2026  

**README de la carpeta `week5/` (objetivo, qué se comparó, resultado principal, cómo ejecutar — rúbrica):** [../README.md](../README.md)

**Notebook:** [`Actividad5_Clasificacion_Multiclase_Keras.ipynb`](Actividad5_Clasificacion_Multiclase_Keras.ipynb)

**Tras publicar en GitHub:** [Abrir notebook en Google Colab](https://colab.research.google.com/github/SonicWD/deep_learning/blob/main/week5/actividad5/Actividad5_Clasificacion_Multiclase_Keras.ipynb)

---

## Objetivo

Implementar una **red neuronal fully connected** para **clasificación multiclase** con **TensorFlow Keras**, y **evidenciar** cómo cambia el entrenamiento al ajustar **hiperparámetros** (tasa de aprendizaje, tamaño de lote, número de épocas) y al cambiar el **optimizador**. Sin capas convolucionales.

**Dataset:** `sklearn.datasets.load_digits` (10 clases, 64 características por imagen 8×8 aplanada). División **train / validación / test** y `StandardScaler` ajustado solo en train.

**Evidencia:** curvas de **loss** y **accuracy** (train vs val), tabla de **accuracy en test**, y **conclusiones breves** comparativas.

---

## Conteo de secciones (notebook)

| # | Sección |
|---|--------|
| 1 | Dataset y preparación |
| 2 | Arquitectura MLP (softmax, `sparse_categorical_crossentropy`) |
| 3 | Comparación de optimizadores (misma red, misma semilla inicial, mismas épocas y batch salvo donde se indique) |
| 4 | Barrido de **tasa de aprendizaje** (Adam) |
| 5 | Barrido de **tamaño de batch** |
| 6 | Tabla de resultados y gráficos |
| 7 | Conclusiones (Markdown) |

---

## Cómo ejecutar

1. Abrir el `.ipynb` en **Google Colab** o Jupyter.
2. **Ejecutar todo** (instala `tensorflow`, `scikit-learn`, `matplotlib`, `numpy` si hace falta).
3. Revisar salidas: figuras y tabla final.

**Nota (PyTorch):** La consigna permite PyTorch; este entregable usa **Keras** por claridad y alineación con Colab. Una versión equivalente en PyTorch sería `nn.Sequential` + `CrossEntropyLoss` + mismo esquema de experimentos.

---

## Qué se compara

| Experimento | Qué varía | Qué se fija (referencia) |
|-------------|-----------|---------------------------|
| Optimizadores | `SGD` (+ momentum), `Adam`, `RMSprop` | Arquitectura, datos, épocas, `batch_size`, semillas |
| Learning rate | Valores típicos por optimizador o barrido con Adam | Resto igual en cada subexperimento |
| Batch size | p. ej. 8, 32, 128 | Mismo optimizador y lr base |

---

*Especialización en Inteligencia Artificial*
