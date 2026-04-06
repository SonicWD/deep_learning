# Semana 4 – Actividad 4: Regularización en redes neuronales

**Curso:** Deep Learning - Conceptos (601539)  
**Actividad:** Aplicación de métodos de regularización en una red neuronal en Google Colab (REA 1)  
**Programa:** FU · CAD2202023205 · EIAIPA2026

---

## Objetivo

Aplicar **métodos de regularización** a una red neuronal para **reducir el sobreajuste (overfitting)** y mejorar la **generalización**. La evidencia consiste en comparar el desempeño **con y sin** regularización, observando el comportamiento del **loss** y las **métricas** en entrenamiento y evaluación, con un notebook **reproducible** y bien documentado.

---

## Qué se implementó

- **Un modelo base (MLP)** entrenable con **backpropagation:** una capa oculta **ReLU**, salida **sigmoide**, pérdida **MSE**.
- **Dos técnicas de regularización desde cero**, integradas en **forward**, **backward** y **actualización de pesos:** **L2 (weight decay)** y **dropout invertido** en la capa oculta.
- **Baseline sin regularización** frente a **solo L2** y **solo dropout**, con la **misma** arquitectura, `lr`, `epochs` y semilla de pesos.
- **Dataset** de clasificación binaria con más características y ruido en etiquetas (`make_classification`), división **train / validación / test** y `StandardScaler`.
- **Curvas** de pérdida (MSE) y **accuracy** (train vs val), **tabla** de métricas (train/val/test), **brecha train–val** y **conclusiones** en Markdown.

---

## Qué técnicas se compararon

| Técnica | Integración | Efecto esperado |
|---------|-------------|-----------------|
| **Ninguna (baseline)** | Sin término extra en gradientes ni máscaras | Mayor riesgo de **sobreajuste** (brecha train–val). |
| **L2** | Suma \((\lambda/n)\,W\) al gradiente de `W1` y `W2` | **Encoge** pesos; suele reducir varianza y estabilizar generalización. |
| **Dropout (invertido)** | Máscara aleatoria en la oculta al entrenar; misma máscara en backward | Obliga a **no depender** de neuronas concretas; regularización estocástica. |

---

## Resultados principales

- Tras **Run all**, el **baseline** suele mostrar **más brecha train–val** en accuracy que los modelos con L2 o dropout.
- **L2** y **dropout** suelen **suavizar** la pérdida en validación o **mejorar** val/test frente al baseline (depende de \(\lambda\), \(p\) y del dataset; el notebook documenta el trade-off).
- Las **figuras** y la **tabla numérica** son la evidencia verificable para la entrega.

---

## Cómo ejecutar el notebook

1. Abre el archivo `Actividad4_Regularizacion_Red_Neuronal.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ejecuta todas las celdas: **Entorno de ejecución → Ejecutar todo** (en inglés: **Runtime → Run all**).
3. Si aparece un mensaje de **instalación de paquetes**, vuelve a ejecutar la celda de imports del dataset hasta que cargue sin error.
4. Revisa las **curvas** (loss y accuracy), la **tabla** train/val/test y las **conclusiones**.

**Dependencias:** `numpy`, `matplotlib`, `scikit-learn` (suelen venir en Colab; en local: `pip install numpy matplotlib scikit-learn`).

---

## 📋 Contenido detallado

El archivo `Actividad4_Regularizacion_Red_Neuronal.ipynb` contiene:

### 1. Dataset y preparación
- Generación con `make_classification` (sklearn): muchas características, `flip_y` para ruido en etiquetas.
- División **train / validación / test** (entrena en train, monitorea en val, evalúa en test).
- Escalado con `StandardScaler` (ajuste solo en train).
- Imprime formas y clases.

### 2. Justificación de las técnicas (Markdown)
- Tabla explicando **por qué L2** y **por qué dropout** en este problema de generalización.

### 3. Red neuronal y regularización
- **Propósito:** Clasificación binaria con una capa oculta.
- **Características:**
  - Activaciones: **ReLU** (oculta), **sigmoide** (salida).
  - Pérdida **MSE**; gradientes en batch completo (un paso por época sobre todo el train).
  - **L2:** penalización sobre `W1` y `W2` en la regla de actualización.
  - **Dropout invertido:** `keep = 1-p`, máscara `/keep`, aplicada a la activación oculta solo en entrenamiento.
  - En validación/test: **sin dropout** (red completa).

### 4. Entrenamiento comparativo
- Tres corridas secuenciales: **baseline**, **L2** (\(\lambda\) configurable), **dropout** (\(p\) configurable).
- Mismos hiperparámetros de red y mismo `seed` para inicialización comparable.

### 5. Gráficos y tabla de resultados
- Pérdida MSE y accuracy frente a la época (train y val) para los tres modelos.
- Impresión de **brecha train–val** (indicador de sobreajuste).

### 6. Conclusiones
- Interpretación del impacto, trade-offs de \(\lambda\) y \(p\), reproducibilidad.

---

## 🚀 Requisitos e instalación

### Dependencias
- Python 3.8+ recomendado
- NumPy
- Matplotlib
- scikit-learn

### Instalación
```bash
pip install numpy matplotlib scikit-learn
```

---

## 📖 Uso en Google Colab

1. Sube el archivo `Actividad4_Regularizacion_Red_Neuronal.ipynb` a [Google Colab](https://colab.research.google.com/) o ábrelo desde GitHub.
2. Ejecuta todas las celdas: **Runtime → Run all** / **Ejecutar todo**.
3. Verifica que salgan la **tabla** de accuracies y las **dos figuras** (loss y accuracy).
4. Descarga el cuaderno ejecutado: **File → Download → .ipynb** (evidencia con salidas).

---

## 🔧 Parámetros principales

| Elemento | Parámetros |
|----------|------------|
| **Red (`MLPBinaria`)** | `n_hidden` (neuronas ocultas), `lr`, `epochs`, `l2_lambda`, `dropout_p`, `seed` |
| **Dataset** | Tamaño de muestra, `n_features`, `flip_y`, `random_state` en `make_classification` |
| **Comparación justa** | Mismos `N_HIDDEN`, `LR`, `EPOCHS`, `SEED` entre baseline, L2 y dropout |

Valores por defecto en el notebook (ajustables): `N_HIDDEN=96`, `LR=0.25`, `EPOCHS=2500`, `L2_LAMBDA=0.08`, `DROPOUT_P=0.35`.

---

## 📊 Criterios de calificación (rúbrica)

### Cumplimiento esperado en esta entrega

| Criterio | Cumplimiento |
|----------|----------------|
| **Implementación de técnicas de regularización** | Al menos **dos** (L2 y dropout) **desde cero**, integradas en forward/backward y actualización de pesos; código ejecutable de principio a fin. |
| **Justificación de la selección** | Markdown **§2** vincula cada técnica con generalización y sobreajuste. |
| **Análisis del impacto** | Curvas de loss/accuracy, métricas train/val/test y texto en **§6** interpretando el efecto. |
| **Claridad y organización** | Clase documentada, celdas en orden: datos → modelo → entrenamiento → figuras → conclusiones. |
| **Comparación con y sin regularización** | **Baseline** explícito vs modelos regularizados; mismos hiperparámetros base; gráficos y tabla. |

### Niveles orientativos (curso)

Cada criterio se califica con una escala típica **0.2 – 1.0** según profundidad:

- **1.0:** Implementación completa, justificación sólida, análisis y comparación claros con métricas y gráficos.
- **0.8:** Cumple bien con detalle mejorable (segunda técnica, visualización o análisis).
- **0.6–0.4:** Integración o evidencia parcial.
- **0.2:** Ausente o incorrecto.

(Ajusta la redacción de conclusiones si tu corrida no muestra mejora en test: explicar por qué — hiperparámetros, dataset, varianza de dropout — también cuenta como análisis válido.)

---

## 📈 Estructura del notebook

1. **Introducción:** título y objetivo de la actividad.
2. **Celda `%pip`:** dependencias rápidas en Colab.
3. **Sección 1:** Dataset (train/val/test, escalado).
4. **Sección 2:** Justificación L2 y dropout (Markdown).
5. **Sección 3:** Definición del MLP + regularización.
6. **Sección 4:** Entrenamiento baseline, L2 y dropout.
7. **Sección 5:** Gráficos comparativos y brecha train–val.
8. **Sección 6:** Conclusiones.

---

## 🎯 Regularización utilizada (resumen)

| Enfoque | Idea clave | Dónde actúa |
|---------|------------|-------------|
| **L2** | Penalizar pesos grandes: gradiente \(\nabla_W J + (\lambda/n) W\) | Pesos `W1`, `W2` |
| **Dropout** | Apagar neuronas al azar en train; escala \(1/(1-p)\) para mantener magnitud | Salida ReLU de la capa oculta (solo train) |

---

## 📝 Notas

- **Loss en las curvas:** se registra **MSE** sobre predicciones para **comparar** los tres modelos en la misma escala (el término L2 se aplica solo en el **gradiente** del entrenamiento, no en el MSE mostrado).
- **Sobreajuste:** una **brecha train–val** grande en accuracy suele indicar memorización; la regularización busca **reducirla** sin destruir el ajuste.
- **Hiperparámetros:** si val/test no mejora, prueba \(\lambda\) más pequeño o \(p\) más bajo antes de concluir que “no sirve” la técnica.

---

## 📚 Referencias

- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press (capítulos sobre regularización).

---

**Curso:** 601539 - Deep Learning - Conceptos | FU | CAD2202023205 | EIAIPA2026
