# Semana 6 – Actividad 6: Métricas, preprocesamiento y regularización (Keras)

**Curso:** 601539 · **Programa:** FU / CAD2202023205 / EIAIPA2026 · **REA 1**

**README de la carpeta `week6/`:** [../README.md](../README.md)

**Notebook:** [`Actividad6_Metricas_Preprocesamiento_Regularizacion_Keras.ipynb`](Actividad6_Metricas_Preprocesamiento_Regularizacion_Keras.ipynb)

**Colab:** [Abrir notebook](https://colab.research.google.com/github/SonicWD/deep_learning/blob/main/week6/actividad6/Actividad6_Metricas_Preprocesamiento_Regularizacion_Keras.ipynb)

---

## Objetivo

Comparar un MLP **sin regularización** frente al **mismo** MLP con **L2** en pesos de capas ocultas y **dropout**, sobre datos con **preprocesamiento** estándar, registrando **loss**, **accuracy**, **AUC** y **precision/recall/F1** en test.

## Método(s) de regularización aplicado(s)

- **Weight decay (L2)** en capas densas ocultas (`kernel_regularizer=l2(1e-3)`).
- **Dropout** (0.35) tras cada capa oculta.

## Comparación realizada (una frase)

Mismo dataset, partición, `StandardScaler`, arquitectura (96→48→1), `Adam` (`lr=1e-3`), 120 épocas y batch 32; solo el modelo regularizado incluye L2 y dropout.

## Resultado principal (típico)

El **base** suele mostrar **mayor brecha train–val** (overfitting); el **regularizado** suele **acercar** validación a entrenamiento y a veces **mejorar** AUC/F1 en test a costa de menor ajuste en train.

## Cómo ejecutar

Google Colab: *Ejecutar todo*. Dependencias: `tensorflow`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`.

## Checklist de entrega final

- Notebook abre en Colab sin errores de dependencias.
- Se ejecutan todas las celdas en orden (de inicio a fin).
- Se evidencia comparación justa: mismos datos/partición/preprocesamiento/hiperparámetros.
- Se incluyen curvas (`loss`, `AUC`) y tabla de métricas (val/test).
- Se redactan conclusiones de overfitting, generalización y trade-offs (sección 6).
- Enlaces de GitHub/Colab funcionando en este README.

---

*Especialización en Inteligencia Artificial*
