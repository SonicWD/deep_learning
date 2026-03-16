# Semana 3 – Actividad 2: Backpropagation y Funciones de Activación

**Curso:** Deep Learning - Conceptos (601539)  
**Actividad:** Implementación de Backpropagation y Funciones de Activación en Redes Neuronales en Google Colab

---

## Objetivo

Implementar y validar el proceso de aprendizaje de redes neuronales mediante **backpropagation** y **funciones de activación**, evidenciando cómo la red ajusta sus parámetros (pesos y sesgos) para reducir el error durante el entrenamiento, y cómo las funciones de activación influyen en el comportamiento del modelo y su capacidad de aprendizaje.

---

## Qué se implementó

- **Tres modelos entrenables con backpropagation:** perceptrón, red de una capa y red multicapa.
- **Dataset de clasificación binaria** (make_classification, 500 muestras, 2 características).
- **Comparación explícita sigmoid vs ReLU** en capas ocultas (mismo modelo, misma configuración).
- **Evolución del loss** durante el entrenamiento y métricas de accuracy (train/test).
- **Conclusiones** sobre variación de activación, comportamiento del loss y hallazgos.

---

## Qué activaciones se compararon

| Activación | Uso | Comparación |
|------------|-----|-------------|
| **Sigmoide** | Salida (todos) y capas ocultas (comparación) | Saturación de gradientes; convergencia más lenta |
| **ReLU** | Capas ocultas (comparación) | Mejor flujo de gradientes; convergencia más rápida |

---

## Resultados principales

- Los tres modelos alcanzan accuracy >90% en train y test.
- ReLU en capas ocultas converge más rápido y con menor pérdida final que sigmoide (comparación directa en Sección 4.5).
- Las curvas de pérdida muestran convergencia consistente en todos los modelos.

---

## Cómo ejecutar el notebook

1. Abre el archivo `Actividad2_Backpropagation_Funciones_Activacion.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ejecuta todas las celdas: **Runtime → Run all**.
3. Revisa las curvas de pérdida, la comparación sigmoid vs ReLU y las conclusiones.

**Dependencias:** `numpy`, `matplotlib`, `scikit-learn` (ya instaladas en Colab; en local: `pip install numpy matplotlib scikit-learn`).

---

## 📋 Contenido detallado

El archivo `Actividad2_Backpropagation_Funciones_Activacion.ipynb` contiene:

### 1. Dataset de clasificación binaria
- Generación con `make_classification` (sklearn)
- 500 muestras, 2 características informativas, 2 clases (0 y 1)
- División train/test (80/20), escalado con `StandardScaler`
- Visualización del dataset

### 2. Modelo 1: Perceptrón con sigmoide y backpropagation
- **Propósito:** Clasificación binaria con una neurona
- **Características:**
  - Función de activación sigmoide en la salida
  - Pérdida MSE (error cuadrático medio)
  - Backpropagation: cálculo de gradientes y actualización de pesos/sesgo
  - Descenso por gradiente

### 3. Modelo 2: Red de una capa
- **Propósito:** Clasificación binaria con capa oculta
- **Características:**
  - Capa oculta con sigmoide
  - Capa de salida con sigmoide (1 neurona)
  - Backpropagation completo en ambas capas
  - Inicialización Xavier

### 4. Modelo 3: Red multicapa
- **Propósito:** Clasificación binaria con múltiples capas ocultas
- **Características:**
  - Capas ocultas con **ReLU**
  - Capa de salida con **sigmoide**
  - Backpropagation completo con derivadas de ReLU
  - Arquitectura configurable (ej: 8-4-1)

### 5. Resumen y conclusiones
- Tabla de accuracy (train/test) para los tres modelos
- Gráficos de pérdida durante el entrenamiento
- Conclusiones breves sustentadas en la evidencia

---

## 🚀 Requisitos e instalación

### Dependencias
- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn

### Instalación
```bash
pip install numpy matplotlib scikit-learn
```

---

## 📖 Uso en Google Colab

1. Sube el archivo `Actividad2_Backpropagation_Funciones_Activacion.ipynb` a [Google Colab](https://colab.research.google.com/).
2. Ejecuta todas las celdas: **Runtime → Run all**.
3. Revisa las curvas de pérdida y las métricas de accuracy.
4. Descarga el cuaderno ejecutado: **File → Download → .ipynb** (para entregar como evidencia).

---

## 🔧 Parámetros principales

| Modelo | Parámetros |
|--------|------------|
| **Perceptrón** | `lr` (tasa de aprendizaje), `epochs` |
| **Red 1 capa** | `n_hidden`, `lr`, `epochs` |
| **Red multicapa** | `hidden` (tupla de neuronas por capa), `lr`, `epochs` |

---

## 📊 Criterios de calificación (rúbrica)

| Criterio | Cumplimiento |
|----------|--------------|
| **Backpropagation** | Implementado correctamente en los 3 modelos; ajustes de pesos eficientes y resultados consistentes. |
| **Funciones de activación** | Sigmoide y ReLU integradas; mejoran la capacidad de clasificación binaria. |
| **Rediseño para clasificación binaria** | Modelos adaptados a dataset binario (etiquetas 0 y 1, una neurona de salida). |
| **Entrenamiento** | Entrenamiento correcto con dataset binario; resultados claros y bien presentados. |
| **Claridad y organización** | Código documentado, comentarios claros, estructura lógica. |

---

## 📈 Estructura del notebook

1. **Sección 1:** Carga y preparación del dataset binario.
2. **Sección 2:** Perceptrón con sigmoide + backprop.
3. **Sección 3:** Red de una capa con sigmoide + backprop.
4. **Sección 4:** Red multicapa con ReLU (oculta) + sigmoide (salida) + backprop.
5. **Sección 4.5:** Comparación sigmoid vs ReLU (mismo modelo, misma config).
6. **Sección 5:** Resumen de resultados y conclusiones.

---

## 🎯 Funciones de activación utilizadas

| Función | Fórmula | Uso |
|---------|---------|-----|
| **Sigmoide** | σ(x) = 1 / (1 + e^(-x)) | Salida (probabilidad 0–1) en todos los modelos |
| **ReLU** | max(0, x) | Capas ocultas en la red multicapa |

---

## 📝 Notas

- **Pérdida:** Se usa MSE para simplificar; en producción suele usarse entropía cruzada binaria (BCE).
- **Convergencia:** La pérdida debe disminuir de forma consistente; si no, ajustar `lr` o `epochs`.
- **Evidencia:** Las curvas de pérdida y las métricas de accuracy sirven como evidencia verificable para la entrega.

---

## 📚 Referencias

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

---

**Curso:** 601539 - Deep Learning - Conceptos | FU | CAD2202023205 | EIAIPA2026
