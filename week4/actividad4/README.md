# Semana 4 – Actividad 4: Regularización en redes neuronales (Google Colab)

**Curso:** Deep Learning - Conceptos (601539)  
**Actividad:** Aplicación de métodos de regularización en una red neuronal en Google Colab (REA 1)  
**Programa:** FU · CAD2202023205 · EIAIPA2026

**Otra entrega de la misma semana:** [Actividad 3 — Optimización](../actividad3/README.md) (SGD vs Adam, mismas condiciones de comparación).

**Archivo principal:** [`Actividad4_Regularizacion_Red_Neuronal.ipynb`](Actividad4_Regularizacion_Red_Neuronal.ipynb) (en esta misma carpeta `week4/actividad4/`).

---

## Objetivo

Aplicar **métodos de regularización** a una red neuronal para **reducir el sobreajuste** y mejorar la **generalización**, con evidencia reproducible: **loss** (MSE) y **accuracy** en entrenamiento, validación y test, más gráficos y una interpretación breve impresa junto a la tabla de resultados.

**Método(s) aplicado(s):** **L2 (weight decay)** sobre `W1` y `W2`, y **dropout invertido** en la capa oculta, ambos integrados en forward, backward y actualización de pesos; más un **baseline sin regularización**.

**Comparación realizada (en una frase):** Se entrenan tres veces el **mismo** MLP sobre el **mismo** dataset particionado y escalado, con **iguales** `n_hidden`, `lr`, `epochs` y semilla de pesos, variando únicamente la presencia de L2 o dropout frente al modelo base.

**Resultado principal:** En una corrida típica el **baseline** muestra mayor **brecha train–val**; **L2** y/o **dropout** suelen **acotar el sobreajuste** y pueden **mejorar o igualar accuracy en validación/test** respecto al base (depende de \(\lambda=0.08\) y \(p=0.35\) fijados en código y del sorteo dropout época a época).

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

## Resultados y evidencia en el notebook

- Tabla **train / val / test** (accuracy y MSE de validación al final) más un bloque **«Interpretación breve»** tras la tabla.
- **Dos figuras:** MSE y accuracy vs época (train y val) para baseline, L2 y dropout.
- Impresión de **brecha train–val** en accuracy al final de la sección de gráficos.
- **Sección 6 (Markdown):** conclusiones en **(i)** overfitting/underfitting, **(ii)** efecto de la regularización, **(iii)** hallazgos y dificultades.

---

## Cómo ejecutar el notebook

1. Abrir `actividad4/Actividad4_Regularizacion_Red_Neuronal.ipynb` en [Google Colab](https://colab.research.google.com/) (desde el repositorio o subiendo el archivo).
2. Ejecutar todas las celdas: **Entorno de ejecución → Ejecutar todo** (**Runtime → Run all**).
3. Si una celda pide **instalación de paquetes**, ejecutar de nuevo la celda de imports del dataset hasta completar la carga.
4. **Salidas:** curvas de pérdida y accuracy (train/val), tabla train/val/test y sección de conclusiones en Markdown.

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

1. Abrir en [Google Colab](https://colab.research.google.com/) el notebook `week4/actividad4/Actividad4_Regularizacion_Red_Neuronal.ipynb` desde GitHub o por carga manual.
2. Ejecutar todas las celdas: **Runtime → Run all** / **Ejecutar todo**.
3. Tras ejecutar todo el cuaderno aparecen la **tabla** de accuracies y **dos figuras** comparativas (loss y accuracy).
4. Opcional: **File → Download → .ipynb** conservando las salidas generadas.

---

## 🔧 Parámetros principales

| Elemento | Parámetros |
|----------|------------|
| **Red (`MLPBinaria`)** | `n_hidden` (neuronas ocultas), `lr`, `epochs`, `l2_lambda`, `dropout_p`, `seed` |
| **Dataset** | Tamaño de muestra, `n_features`, `flip_y`, `random_state` en `make_classification` |
| **Comparación justa** | Mismos `N_HIDDEN`, `LR`, `EPOCHS`, `SEED` entre baseline, L2 y dropout |

Valores por defecto en el notebook (ajustables): `N_HIDDEN=96`, `LR=0.25`, `EPOCHS=2500`, `L2_LAMBDA=0.08`, `DROPOUT_P=0.35`.

---

## Cobertura técnica (resumen)

| Aspecto | Contenido en este repositorio |
|----------|--------------------------------|
| **Regularización** | L2 y dropout invertido en la capa oculta, integrados en forward, backward y actualización de pesos. |
| **Justificación** | Sección 2 (Markdown) enlaza cada técnica con generalización y sobreajuste. |
| **Análisis** | Curvas de loss/accuracy, métricas train/val/test, brecha train–val y conclusiones (sección 6). |
| **Organización** | Flujo: datos → modelo → entrenamiento comparativo → figuras → conclusiones. |
| **Comparación** | Baseline sin regularización frente a L2 y dropout; mismos hiperparámetros base e inicialización. |

*Nota:* Si en una corrida concreta val/test no mejora respecto al baseline, el texto de conclusiones puede centrarse en el efecto observado en la brecha train–val, la sensibilidad a \(\lambda\) y \(p\), y la varianza propia del dropout.

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
- **Hiperparámetros:** con val/test por debajo del esperado, conviene ensayar \(\lambda\) o \(p\) más bajos antes de concluir que la técnica no ayuda en este problema.

---

## 📚 Referencias

- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press (capítulos sobre regularización).

---

**Curso:** 601539 - Deep Learning - Conceptos | FU | CAD2202023205 | EIAIPA2026
