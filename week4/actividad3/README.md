# Semana 4 – Actividad 3: Optimización en redes neuronales (Google Colab)

**Curso:** Deep Learning - Conceptos (601539) · FU · CAD2202023205 · EIAIPA2026 · REA 1  

**Notebook:** [Actividad3_Optimizacion_Red_Neuronal.ipynb](Actividad3_Optimizacion_Red_Neuronal.ipynb)

En la misma semana (REA 1): [Actividad 4 – Regularización](../actividad4/README.md).

---

## Objetivo

Entrenar un perceptrón multicapa para clasificación binaria comparando **SGD en batch completo** con **Adam** (implementación explícita del algoritmo), manteniendo fijos el dataset, la arquitectura, el número de épocas y la semilla de inicialización. La comparación se apoya en curvas de pérdida y accuracy (entrenamiento y validación), tiempos de cómputo y métricas en test.

**Técnicas comparadas:** SGD vanilla (batch completo) frente a **Adam** (momentos con corrección de sesgo).

**Configuración base (en una frase):** MLP de una capa oculta (64 neuronas, ReLU), salida sigmoide, pérdida MSE, 2000 épocas, dataset sintético `make_classification` con partición train/val/test y `StandardScaler` fijos, misma semilla de pesos iniciales en ambas corridas; solo cambia la regla de optimización y sus hiperparámetros asociados (`LR_SGD`, `LR_ADAM`, β₁, β₂, ε).

**Resultado principal:** Con los valores fijados en el notebook (`LR_SGD = 0.25`, `LR_ADAM = 0.01`), **SGD obtiene mejor generalización** que Adam en este problema (del orden de **~0,82** frente a **~0,51** de accuracy en test, y MSE de validación notablemente menor), lo que refuerza que **hay que calibrar el optimizador y la tasa de aprendizaje**; Adam no garantiza por sí solo un mejor resultado.

---

## Contenido del trabajo

- Datos sintéticos (`make_classification`), partición train / validación / test y `StandardScaler`.
- MLP: una capa oculta ReLU, salida sigmoide, pérdida MSE, retropropagación en NumPy.
- Entrenamiento con **SGD vanilla** y con **Adam** (momentos con corrección de sesgo, ε numérico en el denominador).
- Figuras comparativas y resumen impreso (accuracy y pérdida en validación y test, tiempos).

---

## Parámetros documentados en el código

| Parámetro | Uso |
|-----------|-----|
| `LR_SGD` | Tasa de aprendizaje del SGD (batch completo). |
| `LR_ADAM` | Tasa de aprendizaje de Adam (suele ser menor que la del SGD en problemas similares). |
| `BETA1`, `BETA2`, `EPS` | Hiperparámetros estándar de Adam (α adaptativo por parámetro). |
| `N_HIDDEN`, `EPOCHS`, `SEED_INIT` | Arquitectura, presupuesto de épocas y reproducibilidad. |

---

## Cómo ejecutarlo

1. Subir el `.ipynb` a [Google Colab](https://colab.research.google.com/) o abrirlo en Jupyter/VS Code con kernel Python 3.
2. **Ejecutar todas las celdas** (en Colab: Entorno → Ejecutar todo).
3. La primera celda instala `numpy`, `matplotlib` y `scikit-learn` si hace falta (`%pip`).

Dependencias: `numpy`, `matplotlib`, `scikit-learn`.

```bash
pip install numpy matplotlib scikit-learn
```

---

## Referencias

Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.*  
Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press (capítulo de optimización).

---

*Especialización en Inteligencia Artificial — Deep Learning*
