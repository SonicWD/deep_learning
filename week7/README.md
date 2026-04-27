# Semana 7 — Actividad 7: Convolución manual, padding y stride

**Asignatura / ruta sugerida:** Deep Learning (REA 1) · Evidencia en GitHub en `week7/`.

## Objetivo

Implementar la operación de **convolución 2D** (correlación cruzada, estilo CNN) con **NumPy** sobre matrices e imágenes en escala de gris, mostrando el efecto de **padding** y **stride** con el **mismo kernel 3×3** en todos los casos comparables.

## Qué se comparó (rúbrica)

| Caso | Padding | Stride | Idea |
|------|---------|--------|------|
| **A** (referencia) | 0 | 1 | Mapa base del filtro Laplaciano. |
| **B** (solo padding) | 2 | 1 | Mayor tamaño de salida; más posiciones en bordes. |
| **C** (solo stride) | 0 | 2 | Mapa más pequeño (submuestreo espacial). |

**Kernel fijo:** matriz Laplaciana 3×3 (`K_LAP` en el notebook).

## Evidencia principal

- Salida numérica en matriz 5×5 y figuras con imagen sintética 64×64 e imagen **camera** (o patrón sintético si no hay `scikit-image`), con normalización para visualizar el mapa de respuesta.

## Cómo ejecutar (Google Colab)

1. Sube `Actividad7_Convolucion_Padding_Stride.ipynb` a [Colab](https://colab.research.google.com).
2. En la **primera celda de código**, si hace falta, descomenta:  
   `!pip install -q numpy matplotlib scikit-image`  
   y vuelve a ejecutar **Run all**.
3. Revisa que no haya errores; las figuras deben mostrarse al final de las secciones 3 y 3.1.

## Cómo ejecutar (local)

```bash
cd deep_learning/week7
python -m pip install numpy matplotlib scikit-image jupyter
jupyter notebook Actividad7_Convolucion_Padding_Stride.ipynb
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `Actividad7_Convolucion_Padding_Stride.ipynb` | Notebook único con implementación, visualizaciones y **3 conclusiones** al cierre. |

## Conclusiones (resumen; el detalle está en el notebook)

1. La correlación 2D manual coincide con suma de productos por ventana (verificación en 5×5).  
2. Aumentar **padding** con **stride 1** incrementa dimensiones de salida y el rol de los bordes.  
3. Aumentar **stride** reduce resolución espacial del mapa sin cambiar el kernel.

*Fechas de la actividad (plataforma): apertura 13-feb-2026, cierre 26-abr-2026, 23:59.*
