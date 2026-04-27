# Semana 8 — Actividad 8: CNN + Transfer Learning (TensorFlow / Keras)

**REA 1 · Deep Learning** · Evidencia en GitHub en `week8/`.

## Objetivo

Implementar un flujo de **clasificación de imágenes** con:

1. Una **CNN entrenada desde cero** sobre un subconjunto **limitado** de CIFAR-10.
2. Un modelo de **transfer learning** con **MobileNetV2** (ImageNet) y base **congelada** + cabezal denso.

Misma partición train/val y mismo conjunto de **test** para comparación controlada.

## Dataset (términos generales)

**CIFAR-10:** 60.000 imágenes de entrenamiento y 10.000 de prueba, 32×32 píxeles, RGB, 10 clases. En el notebook se usa un subconjunto fijo del entrenamiento (`N_LIMIT = 8000`) para simular **datos limitados** en ambos modelos.

## Arquitectura (términos generales)

| Enfoque | Descripción breve |
|--------|-------------------|
| **Modelo A** | Bloques convolucionales + *max pooling* + `GlobalAveragePooling2D` + `Dense` softmax. |
| **Modelo B** | **MobileNetV2** preentrenado (base no entrenable) + cabezal de clasificación; preprocesado `preprocess_input` de MobileNet. |

## Comparación realizada

Mismas muestras de entrenamiento/validación; **misma métrica** en *test* (pérdida y **accuracy**). Gráficas de entrenamiento y **matrices de confusión** por modelo. Barra comparativa de *test accuracy*.

## Resultado principal (referencia)

Depende de la ejecución; el notebook imprime `test loss` y `test accuracy` para **scratch** y **transfer**. Suele observarse mejor uso de datos con la base preentrenada cuando el entrenamiento es corto y el subconjunto es pequeño (no garantizado en todos los runs).

## Cómo ejecutar (Google Colab)

1. Subir `Actividad8_CNN_TransferLearning.ipynb` a [Colab](https://colab.research.google.com).
2. Opcional: **Entorno de ejecución → Acelerador GPU** (más rápido).
3. Si falla TensorFlow: descomente en la celda 0 `!pip install -q "tensorflow>=2.14"`.
4. **Entorno de ejecución → Ejecutar todo**.

## Cómo ejecutar (local)

**Importante (Windows):** con **Python 3.13** TensorFlow a veces no instala correctamente o falla con `No module named 'tensorflow.python'`. Se recomienda **Python 3.11 o 3.12** en un entorno virtual, o **Google Colab**.

```bash
cd deep_learning/week8
python -m venv .venv
.venv\Scripts\activate
python -m pip install tensorflow scikit-learn matplotlib seaborn jupyter
jupyter notebook Actividad8_CNN_TransferLearning.ipynb
```

En la primera celda del notebook hay un `try/except` que, si el import falla, indica pasos (Colab, venv, reinstalación).

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `Actividad8_CNN_TransferLearning.ipynb` | Pipeline, entrenamiento, evaluación, comparación y **5 conclusiones** en Markdown. |

**Cierre de plataforma (según consigna):** 26 de abril de 2026, 23:59.
