# Semana 4 – Actividad 3: Optimización en redes neuronales

**Curso:** Deep Learning - Conceptos (601539)  
**Actividad:** Aplicación de técnicas de optimización en una red neuronal en Google Colab  
**Programa:** FU · CAD2202023205 · EIAIPA2026 · **REA 1**

**Notebook:** [`Actividad3_Optimizacion_Red_Neuronal.ipynb`](Actividad3_Optimizacion_Red_Neuronal.ipynb) — SGD vanilla vs **Adam** desde cero; mismos pesos iniciales y épocas; curvas, métricas y tiempos.

**Misma semana (REA 1):** [Actividad 4 — Regularización](../actividad4/README.md) (L2 y dropout; notebook en `actividad4/`).

---

## Objetivo

Aplicar **técnicas de optimización** al entrenamiento de una red neuronal en **Google Colab** para entender **cómo cambia el aprendizaje** cuando se modifica la forma de **actualizar los parámetros** (pesos y sesgos). La optimización debe **evidenciarse** en la **evolución del entrenamiento** (comportamiento del **loss** y/o **métricas**) y en una **comparación explícita** entre configuraciones (por ejemplo: descenso por gradiente clásico vs un optimizador más avanzado).

La actividad refuerza criterio sobre **hiperparámetros** (p. ej. **tasa de aprendizaje**) y sobre **optimizadores** (variantes del descenso por gradiente), manteniendo **trazabilidad** y **evidencia reproducible** en un notebook.

---

## Qué debes implementar y entregar

- **Red neuronal entrenable** (puede ser un MLP para clasificación binaria, como en semanas previas) con **backpropagation** correcto.
- **Al menos una técnica de optimización** implementada **desde cero** o claramente integrada en tu bucle de entrenamiento (no basta con llamar a un wrapper opaco sin explicar las actualizaciones). Ejemplos típicos:
  - **SGD + momentum** (acumulador de velocidad \(v\), actualización \(W \leftarrow W - \alpha v\)).
  - **RMSprop** (media móvil del cuadrado de gradientes, paso adaptativo por parámetro).
  - **Adam** (estimadores de primer y segundo momento; sesgo corregido opcional pero recomendable).
- **Línea base comparable:** mismo **dataset**, **arquitectura** y **número de épocas** (o mismo presupuesto de pasos), usando **SGD sin momentum** (o el optimizador “vanilla” más simple que ya tengas) como **antes**.
- **Después:** la **misma** red con la **técnica de optimización elegida**, registrando **loss** (y accuracy u otra métrica) en train y, si aplica, validación.
- **Justificación en Markdown** de por qué esa técnica es adecuada para tu problema (curvatura, ruido en gradientes, convergencia, etc.).
- **Análisis:** cómo cambió la curva de pérdida, la métrica final, la estabilidad y, si lo miden, **épocas hasta convergencia** o tiempo por época.

---

## Comparación sugerida (antes vs después)

| Configuración | Rol | Qué fijar igual |
|---------------|-----|------------------|
| **Baseline** | SGD / descenso por gradiente simple, un paso global `lr` | Datos, init de pesos (misma semilla), arquitectura, `epochs` |
| **Optimizado** | Momentum, RMSprop o Adam (una elección bien explicada) | Todo lo anterior; solo cambia la **regla de actualización** |

Así la comparación es **válida**: lo único que cambia es el **optimizador** (y sus hiperparámetros, documentados).

---

## Cómo ejecutar el notebook

1. Abre `week4/actividad3/Actividad3_Optimizacion_Red_Neuronal.ipynb` en [Google Colab](https://colab.research.google.com/).
2. **Runtime → Run all** / **Ejecutar todo**.
3. Comprueba que las **curvas** (loss y, si aplica, métrica) y la **tabla** baseline vs optimizado tengan salida numérica.
4. Para entrega: **File → Download → .ipynb** con celdas ejecutadas.

**Dependencias:** `numpy`, `matplotlib`, `scikit-learn` (datasets, splits, métricas opcionales).

```bash
pip install numpy matplotlib scikit-learn
```

---

## 📋 Contenido sugerido del notebook

### 1. Dataset y preparación
- Datos binarios (`make_classification` o similar), train/(val)/test, `StandardScaler` si corresponde.

### 2. Modelo
- Forward, loss (p. ej. MSE o BCE), backward, cálculo de gradientes \(\nabla_W J\), \(\nabla_b J\).

### 3. Baseline: entrenamiento con SGD vanilla
- Actualización: \(W \leftarrow W - \alpha \nabla_W J\) (y análogo para \(b\)).
- Historial: `train_loss`, opcional `val_loss` y accuracy.

### 4. Mismo modelo con la técnica elegida
- Implementación paso a paso del optimizador (estado: \(v\), \(m\), \(s\), etc.).
- Mismas épocas e inicialización (misma `seed` que el baseline).

### 5. Figuras y tabla
- Loss vs época (baseline vs optimizado en el mismo gráfico o subplots).
- Métrica final y, si aplica, breve comentario sobre oscilaciones o velocidad de descenso.

### 6. Justificación y conclusiones (Markdown)
- Por qué elegiste ese optimizador.
- Qué mejoró o empeoró (y con qué `lr`, \(\beta\), etc.).

---

## 🔧 Parámetros a documentar

| Parámetro | Significado |
|------------|-------------|
| `lr` / \(\alpha\) | Tasa de aprendizaje (puede necesitar ajuste distinta entre SGD y Adam). |
| Momentum \(\beta\) | Si usas SGD+momentum (típ. 0.9). |
| RMSprop / Adam | \(\beta_1\), \(\beta_2\), \(\epsilon\) (epsilon de estabilidad numérica). |
| `epochs`, `batch_size` | Si usas mini-batch, documentar tamaño de batch (full batch es más simple de implementar). |

---

## 📊 Criterios de calificación (rúbrica)

### Cumplimiento esperado en esta entrega

| Criterio | Qué evidenciar |
|----------|----------------|
| **Implementación de técnicas de optimización** | Optimizador **correctamente integrado** en el entrenamiento; código que **ejecuta** y muestra mejora razonable o análisis si no hay mejora (p. ej. `lr` desbalanceado). |
| **Comparación antes / después** | **Baseline** vs **optimizado**, mismas bases; gráficos y/o tabla con loss y métricas. |
| **Justificación de la técnica** | Markdown que explique la elección (problema, escala de gradientes, convergencia). |
| **Claridad y organización** | Comentarios por bloques; variables y funciones con nombres claros; orden lógico de celdas. |
| **Análisis del impacto** | Texto que interprete **loss**, **precisión**, **estabilidad** y, si lo miden, **tiempo** o épocas hasta umbral de pérdida. |

### Niveles orientativos (escala típica 0.2 – 1.0 por criterio)

- **1.0:** Implementación sólida, comparación clara, justificación y análisis bien fundamentados.
- **0.8:** Cumple bien; detalle o eficiencia mejorable.
- **0.6 – 0.4:** Implementación parcial, comparación o análisis débiles.
- **0.2:** No entregado, incorrecto o sin evidencia válida.

---

## 📈 Estructura resumida del notebook

1. Introducción y objetivo.  
2. Datos.  
3. Red + pérdida + gradientes.  
4. Entrenamiento baseline (SGD).  
5. Entrenamiento con optimizador elegido.  
6. Gráficas y tabla comparativa.  
7. Justificación y conclusiones.

---

## 🎯 Ideas de optimizadores (recordatorio breve)

| Método | Idea |
|--------|------|
| **SGD** | Paso proporcional al gradiente negativo. |
| **Momentum** | Suaviza direcciones acumulando velocidad; ayuda en valles y curvas alargadas. |
| **RMSprop** | Paso adaptativo por parámetro usando media del **cuadrado** de gradientes recientes. |
| **Adam** | Combina momentos de primer y segundo orden; muy usado por defecto en muchos problemas. |

---

## 📝 Notas

- Un mismo **Adam** con `lr` demasiado alto puede **diverger**; para comparar con SGD suele hacer falta **buscar** `lr` razonable para cada método.
- Reutilizar la **misma semilla** para inicialización de pesos entre corridas hace la comparación más **justa**.
- Si usas **mini-batch**, el gradiente es estocástico: las curvas pueden ser más ruidosas que en batch completo.

---

## 📚 Referencias

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.  
- Hinton, G. (cursos/notas). RMSprop (describir en notas de optimización).  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press (capítulo de optimización).

---

**Curso:** 601539 – Deep Learning – Conceptos | FU | CAD2202023205 | EIAIPA2026

*Especialización en Inteligencia Artificial — Deep Learning*
