# Semana 2 – Redes neuronales básicas (NumPy)

Implementación de tres arquitecturas desde cero con NumPy: perceptrón, red de una capa y red multicapa.

## 📋 Contenido

El archivo `ejercicio1.py` contiene las siguientes implementaciones:

### 1. Perceptrón Simple (`Perceptron`)
- **Propósito**: Clasificación binaria de problemas linealmente separables
- **Características**:
  - Función de activación escalón (step function)
  - Algoritmo de aprendizaje supervisado
  - Actualización de pesos mediante regla de aprendizaje del perceptrón
  - Historial de errores durante el entrenamiento

### 2. Red Neuronal de Una Capa (`SingleLayerNeuralNetwork`)
- **Propósito**: Clasificación multiclase con una capa de salida
- **Características**:
  - Función de activación sigmoide
  - Múltiples neuronas en la capa de salida
  - Backpropagation básico
  - Soporte para clasificación multiclase mediante one-hot encoding

### 3. Red Neuronal Multicapa (`MultiLayerNeuralNetwork`)
- **Propósito**: Resolver problemas no linealmente separables
- **Características**:
  - Múltiples capas ocultas configurables
  - Múltiples funciones de activación: sigmoide, tanh, ReLU
  - Backpropagation completo
  - Inicialización de pesos optimizada (Xavier/He)

## 🚀 Instalación y Requisitos

### Requisitos del Sistema
- Python 3.7 o superior
- NumPy
- Matplotlib (para visualizaciones opcionales)
- scikit-learn (para generar datos de ejemplo)

### Instalación de Dependencias

```bash
pip install numpy matplotlib scikit-learn
```

## 📖 Uso

### Ejemplo 1: Perceptrón para Clasificación Binaria

```python
from ejercicio1 import Perceptron, generate_linearly_separable_data

# Generar datos linealmente separables
X, y = generate_linearly_separable_data(n_samples=100)

# Crear y entrenar el perceptrón
perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)

# Realizar predicciones
predictions = perceptron.predict(X)

# Calcular precisión
accuracy = perceptron.score(X, y)
print(f"Precisión: {accuracy:.2%}")
```

### Ejemplo 2: Red Neuronal de Una Capa para Clasificación Multiclase

```python
from ejercicio1 import SingleLayerNeuralNetwork
from sklearn.datasets import make_classification

# Generar datos multiclase
X, y = make_classification(n_samples=200, n_features=4, 
                          n_classes=3, random_state=42)

# Crear y entrenar la red
slnn = SingleLayerNeuralNetwork(learning_rate=0.1, 
                                n_iterations=500, 
                                n_neurons=3)
slnn.fit(X, y)

# Evaluar
accuracy = slnn.score(X, y)
print(f"Precisión: {accuracy:.2%}")
```

### Ejemplo 3: Red Neuronal Multicapa para Problemas No Lineales

```python
from ejercicio1 import MultiLayerNeuralNetwork, generate_xor_data

# Generar datos XOR (no linealmente separable)
X, y = generate_xor_data(n_samples=100)

# Crear red multicapa con 2 capas ocultas de 5 neuronas cada una
mlp = MultiLayerNeuralNetwork(learning_rate=0.1, 
                              n_iterations=1000,
                              hidden_layers=[5, 5], 
                              activation='tanh')
mlp.fit(X, y)

# Evaluar
accuracy = mlp.score(X, y)
print(f"Precisión: {accuracy:.2%}")
```

## 🔧 Parámetros Principales

### Perceptron
- `learning_rate` (float): Tasa de aprendizaje (default: 0.01)
- `n_iterations` (int): Número máximo de iteraciones (default: 1000)

### SingleLayerNeuralNetwork
- `learning_rate` (float): Tasa de aprendizaje (default: 0.01)
- `n_iterations` (int): Número máximo de iteraciones (default: 1000)
- `n_neurons` (int): Número de neuronas en la capa de salida (default: 1)

### MultiLayerNeuralNetwork
- `learning_rate` (float): Tasa de aprendizaje (default: 0.01)
- `n_iterations` (int): Número máximo de iteraciones (default: 1000)
- `hidden_layers` (list): Lista con número de neuronas por capa oculta (default: [5])
- `activation` (str): Función de activación - 'sigmoid', 'tanh', 'relu' (default: 'sigmoid')

## 📊 Estructura del Código

### Operaciones con NumPy

Todas las operaciones matriciales están implementadas usando NumPy para máxima eficiencia:

- **Multiplicación de matrices**: `np.dot()` para forward propagation
- **Operaciones elemento a elemento**: Para funciones de activación y sus derivadas
- **Transposiciones**: `X.T` para backpropagation
- **Operaciones de suma**: `np.sum()` con `axis` para agregaciones
- **Broadcasting**: Para operaciones con bias

### Funciones de Activación Implementadas

1. **Sigmoide**: `σ(x) = 1 / (1 + e^(-x))`
   - Rango: (0, 1)
   - Útil para clasificación binaria y multiclase

2. **Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
   - Rango: (-1, 1)
   - Útil para capas ocultas

3. **ReLU**: `ReLU(x) = max(0, x)`
   - Rango: [0, ∞)
   - Útil para redes profundas

## 🎯 Características Destacadas

### ✅ Implementación Completa
- **Perceptrón**: Resuelve correctamente problemas linealmente separables
- **Red de una capa**: Maneja clasificación multiclase eficientemente
- **Red multicapa**: Resuelve problemas no lineales complejos (ej: XOR)

### ✅ Uso Eficiente de NumPy
- Todas las operaciones matriciales optimizadas
- Vectorización completa de cálculos
- Sin bucles innecesarios en operaciones críticas

### ✅ Documentación Completa
- Docstrings detallados para todas las clases y métodos
- Comentarios explicativos en código complejo
- Ejemplos de uso incluidos

### ✅ Organización del Código
- Código modular y bien estructurado
- Separación clara entre clases
- Funciones auxiliares para ejemplos y visualización

## 📈 Visualización (Opcional)

El código incluye una función para visualizar fronteras de decisión:

```python
from ejercicio1 import plot_decision_boundary

# Después de entrenar un modelo
plot_decision_boundary(model, X, y, title="Frontera de Decisión")
```

## 🧪 Ejecutar Ejemplos

Para ejecutar los ejemplos incluidos en el archivo:

```bash
python ejercicio1.py
```

Esto ejecutará tres ejemplos:
1. Perceptrón con datos linealmente separables
2. Red de una capa con datos multiclase
3. Red multicapa con problema XOR

## 📝 Notas Importantes

1. **Normalización de Datos**: Para mejores resultados, se recomienda normalizar los datos de entrada antes del entrenamiento.

2. **Tasa de Aprendizaje**: Valores muy altos pueden causar inestabilidad, valores muy bajos pueden hacer el entrenamiento muy lento.

3. **Inicialización de Pesos**: Las redes multicapa usan inicialización de Xavier/He para evitar problemas de gradientes que desaparecen o explotan.

4. **Convergencia**: El perceptrón converge cuando encuentra una solución (errores = 0). Las redes neuronales pueden necesitar ajuste de hiperparámetros.

## Extensiones posibles

1. Variar la tasa de aprendizaje y registrar el efecto en la convergencia.  
2. Comparar funciones de activación (sigmoide, tanh, ReLU) en la misma tarea.  
3. Probar distintas profundidades y anchos de capas ocultas.  
4. Repetir con otros datasets de `sklearn` (por ejemplo Iris, Wine).

## 📚 Referencias

- Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

## 👤 Autor

Implementado como parte del curso de Especialización en IA - Deep Learning Semana 1.

---

**Nota**: Esta implementación es educativa y está diseñada para entender los fundamentos de las redes neuronales. Para aplicaciones de producción, se recomienda usar frameworks especializados como TensorFlow o PyTorch.

