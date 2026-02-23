"""
Actividad: Implementación de Redes Neuronales Básicas
=====================================================

Este módulo contiene la implementación de tres tipos de redes neuronales:
1. Perceptrón simple
2. Red neuronal de una capa (Single Layer Neural Network)
3. Red neuronal multicapa (Multi-Layer Neural Network)

Todas las implementaciones utilizan NumPy para operaciones matriciales eficientes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


# ============================================================================
# 1. PERCEPTRÓN SIMPLE
# ============================================================================

class Perceptron:
    """
    Implementación de un perceptrón simple para clasificación binaria.
    
    El perceptrón es un algoritmo de aprendizaje supervisado que puede
    resolver problemas de clasificación linealmente separables.
    
    Atributos:
        learning_rate (float): Tasa de aprendizaje (default: 0.01)
        n_iterations (int): Número de iteraciones de entrenamiento (default: 1000)
        weights (np.ndarray): Pesos del perceptrón
        bias (float): Sesgo (bias) del perceptrón
        errors_ (list): Historial de errores en cada iteración
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Inicializa el perceptrón.
        
        Args:
            learning_rate: Tasa de aprendizaje (eta)
            n_iterations: Número máximo de iteraciones
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Entrena el perceptrón con los datos de entrada.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            y: Vector de etiquetas binarias (-1 o 1) (n_samples,)
            
        Returns:
            self: Instancia del perceptrón entrenado
        """
        # Inicializar pesos y bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Asegurar que las etiquetas sean -1 o 1
        y = np.where(y == 0, -1, y)
        
        # Entrenamiento
        for _ in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                # Calcular la salida del perceptrón
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = self._activation(linear_output)
                
                # Actualizar pesos si hay error
                if prediction != y[idx]:
                    update = self.learning_rate * y[idx]
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.errors_.append(errors)
            
            # Si no hay errores, el perceptrón convergió
            if errors == 0:
                break
        
        return self
    
    def _activation(self, x: float) -> int:
        """
        Función de activación del perceptrón (función escalón).
        
        Args:
            x: Valor de entrada
            
        Returns:
            1 si x >= 0, -1 en caso contrario
        """
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para nuevos datos.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            
        Returns:
            Vector de predicciones (-1 o 1)
        """
        if self.weights is None:
            raise ValueError("El perceptrón debe ser entrenado antes de predecir")
        
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la precisión del perceptrón.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas verdaderas
            
        Returns:
            Precisión (0.0 a 1.0)
        """
        y = np.where(y == 0, -1, y)
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 2. RED NEURONAL DE UNA CAPA
# ============================================================================

class SingleLayerNeuralNetwork:
    """
    Red neuronal de una capa con múltiples neuronas.
    
    Esta red puede manejar problemas de clasificación multiclase usando
    múltiples neuronas en la capa de salida.
    
    Atributos:
        learning_rate (float): Tasa de aprendizaje
        n_iterations (int): Número de iteraciones
        n_neurons (int): Número de neuronas en la capa de salida
        weights (np.ndarray): Matriz de pesos (n_features, n_neurons)
        bias (np.ndarray): Vector de sesgos (n_neurons,)
        errors_ (list): Historial de errores
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 n_neurons: int = 1):
        """
        Inicializa la red neuronal de una capa.
        
        Args:
            learning_rate: Tasa de aprendizaje
            n_iterations: Número máximo de iteraciones
            n_neurons: Número de neuronas en la capa de salida
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None
        self.errors_ = []
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Función de activación sigmoide.
        
        Args:
            x: Valores de entrada
            
        Returns:
            Valores activados usando sigmoide
        """
        # Evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivada de la función sigmoide.
        
        Args:
            x: Valores de entrada
            
        Returns:
            Derivada de la sigmoide
        """
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SingleLayerNeuralNetwork':
        """
        Entrena la red neuronal de una capa.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            y: Vector de etiquetas (n_samples,) o matriz one-hot (n_samples, n_classes)
            
        Returns:
            self: Instancia entrenada
        """
        n_samples, n_features = X.shape
        
        # Convertir y a formato one-hot si es necesario
        if y.ndim == 1:
            n_classes = len(np.unique(y))
            y_one_hot = np.zeros((n_samples, n_classes))
            for i, label in enumerate(y):
                y_one_hot[i, label] = 1
            y = y_one_hot
        
        n_classes = y.shape[1]
        self.n_neurons = n_classes
        
        # Inicializar pesos y bias usando inicialización de Xavier
        self.weights = np.random.randn(n_features, n_classes) * np.sqrt(2.0 / n_features)
        self.bias = np.zeros((1, n_classes))
        
        # Entrenamiento
        for iteration in range(self.n_iterations):
            # Forward propagation
            linear_output = np.dot(X, self.weights) + self.bias
            output = self._sigmoid(linear_output)
            
            # Calcular error
            error = y - output
            total_error = np.mean(np.abs(error))
            self.errors_.append(total_error)
            
            # Backward propagation
            delta = error * self._sigmoid_derivative(linear_output)
            
            # Actualizar pesos y bias
            self.weights += self.learning_rate * np.dot(X.T, delta)
            self.bias += self.learning_rate * np.sum(delta, axis=0, keepdims=True)
            
            # Mostrar progreso cada 100 iteraciones
            if (iteration + 1) % 100 == 0:
                print(f"Iteración {iteration + 1}/{self.n_iterations}, Error: {total_error:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para nuevos datos.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            
        Returns:
            Vector de predicciones (n_samples,)
        """
        if self.weights is None:
            raise ValueError("La red debe ser entrenada antes de predecir")
        
        linear_output = np.dot(X, self.weights) + self.bias
        output = self._sigmoid(linear_output)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las probabilidades de cada clase.
        
        Args:
            X: Matriz de características
            
        Returns:
            Matriz de probabilidades (n_samples, n_classes)
        """
        if self.weights is None:
            raise ValueError("La red debe ser entrenada antes de predecir")
        
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la precisión de la red.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas verdaderas
            
        Returns:
            Precisión (0.0 a 1.0)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 3. RED NEURONAL MULTICAPA
# ============================================================================

class MultiLayerNeuralNetwork:
    """
    Red neuronal multicapa (MLP) con backpropagation.
    
    Esta red puede tener múltiples capas ocultas y diferentes funciones
    de activación en cada capa.
    
    Atributos:
        learning_rate (float): Tasa de aprendizaje
        n_iterations (int): Número de iteraciones
        hidden_layers (list): Lista con el número de neuronas por capa oculta
        activation (str): Función de activación ('sigmoid', 'tanh', 'relu')
        weights (list): Lista de matrices de pesos para cada capa
        biases (list): Lista de vectores de sesgos para cada capa
        errors_ (list): Historial de errores
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 hidden_layers: List[int] = [5], activation: str = 'sigmoid'):
        """
        Inicializa la red neuronal multicapa.
        
        Args:
            learning_rate: Tasa de aprendizaje
            n_iterations: Número máximo de iteraciones
            hidden_layers: Lista con número de neuronas por capa oculta
            activation: Función de activación ('sigmoid', 'tanh', 'relu')
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.weights = []
        self.biases = []
        self.errors_ = []
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función de activación sigmoide."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de la sigmoide."""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Función de activación tanh."""
        return np.tanh(x)
    
    def _tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de tanh."""
        return 1 - np.tanh(x) ** 2
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Función de activación ReLU."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ReLU."""
        return (x > 0).astype(float)
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica la función de activación según el tipo especificado.
        
        Args:
            x: Valores de entrada
            
        Returns:
            Valores activados
        """
        if self.activation == 'sigmoid':
            return self._sigmoid(x)
        elif self.activation == 'tanh':
            return self._tanh(x)
        elif self.activation == 'relu':
            return self._relu(x)
        else:
            raise ValueError(f"Función de activación '{self.activation}' no soportada")
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica la derivada de la función de activación.
        
        Args:
            x: Valores de entrada
            
        Returns:
            Derivadas
        """
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return self._tanh_derivative(x)
        elif self.activation == 'relu':
            return self._relu_derivative(x)
        else:
            raise ValueError(f"Función de activación '{self.activation}' no soportada")
    
    def _initialize_weights(self, n_features: int, n_classes: int):
        """
        Inicializa los pesos y sesgos de todas las capas.
        
        Args:
            n_features: Número de características de entrada
            n_classes: Número de clases de salida
        """
        self.weights = []
        self.biases = []
        
        # Construir arquitectura completa
        layer_sizes = [n_features] + self.hidden_layers + [n_classes]
        
        # Inicializar pesos para cada capa
        for i in range(len(layer_sizes) - 1):
            # Inicialización de Xavier/He
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(2.0 / (fan_in + fan_out))
            
            w = np.random.randn(fan_in, fan_out) * limit
            b = np.zeros((1, fan_out))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiLayerNeuralNetwork':
        """
        Entrena la red neuronal multicapa usando backpropagation.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            y: Vector de etiquetas (n_samples,) o matriz one-hot
            
        Returns:
            self: Instancia entrenada
        """
        n_samples, n_features = X.shape
        
        # Convertir y a formato one-hot si es necesario
        if y.ndim == 1:
            n_classes = len(np.unique(y))
            y_one_hot = np.zeros((n_samples, n_classes))
            for i, label in enumerate(y):
                y_one_hot[i, label] = 1
            y = y_one_hot
        else:
            n_classes = y.shape[1]
        
        # Inicializar pesos
        self._initialize_weights(n_features, n_classes)
        
        # Entrenamiento
        for iteration in range(self.n_iterations):
            # Forward propagation
            activations = [X]  # Guardar activaciones de cada capa
            z_values = []      # Guardar valores antes de activación
            
            current_input = X
            for i in range(len(self.weights)):
                z = np.dot(current_input, self.weights[i]) + self.biases[i]
                z_values.append(z)
                
                # Usar sigmoide en la última capa para clasificación
                if i == len(self.weights) - 1:
                    a = self._sigmoid(z)
                else:
                    a = self._activate(z)
                
                activations.append(a)
                current_input = a
            
            output = activations[-1]
            
            # Calcular error
            error = y - output
            total_error = np.mean(np.abs(error))
            self.errors_.append(total_error)
            
            # Backward propagation
            deltas = []
            
            # Delta de la capa de salida
            delta = error * self._sigmoid_derivative(z_values[-1])
            deltas.insert(0, delta)
            
            # Deltas de las capas ocultas (propagación hacia atrás)
            for i in range(len(self.weights) - 2, -1, -1):
                delta = np.dot(deltas[0], self.weights[i + 1].T) * self._activate_derivative(z_values[i])
                deltas.insert(0, delta)
            
            # Actualizar pesos y sesgos
            for i in range(len(self.weights)):
                self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
                self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
            
            # Mostrar progreso cada 100 iteraciones
            if (iteration + 1) % 100 == 0:
                print(f"Iteración {iteration + 1}/{self.n_iterations}, Error: {total_error:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones para nuevos datos.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            
        Returns:
            Vector de predicciones (n_samples,)
        """
        if len(self.weights) == 0:
            raise ValueError("La red debe ser entrenada antes de predecir")
        
        # Forward propagation
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._activate(z)
            
            current_input = a
        
        return np.argmax(current_input, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las probabilidades de cada clase.
        
        Args:
            X: Matriz de características
            
        Returns:
            Matriz de probabilidades (n_samples, n_classes)
        """
        if len(self.weights) == 0:
            raise ValueError("La red debe ser entrenada antes de predecir")
        
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._activate(z)
            
            current_input = a
        
        return current_input
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la precisión de la red.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas verdaderas
            
        Returns:
            Precisión (0.0 a 1.0)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# FUNCIONES AUXILIARES PARA EJEMPLOS Y VISUALIZACIÓN
# ============================================================================

def generate_linearly_separable_data(n_samples: int = 100, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos linealmente separables para probar el perceptrón.
    
    Args:
        n_samples: Número de muestras
        random_state: Semilla para reproducibilidad
        
    Returns:
        X: Matriz de características
        y: Vector de etiquetas
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    return X, y


def generate_xor_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos del problema XOR (no linealmente separable).
    
    Args:
        n_samples: Número de muestras por clase
        
    Returns:
        X: Matriz de características
        y: Vector de etiquetas
    """
    np.random.seed(42)
    
    # Clase 0: (0,0) y (1,1)
    class0_1 = np.random.randn(n_samples // 2, 2) * 0.2
    class0_2 = np.random.randn(n_samples // 2, 2) * 0.2 + np.array([1, 1])
    
    # Clase 1: (0,1) y (1,0)
    class1_1 = np.random.randn(n_samples // 2, 2) * 0.2 + np.array([0, 1])
    class1_2 = np.random.randn(n_samples // 2, 2) * 0.2 + np.array([1, 0])
    
    X = np.vstack([class0_1, class0_2, class1_1, class1_2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)]).astype(int)
    
    return X, y


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary"):
    """
    Visualiza la frontera de decisión del modelo.
    
    Args:
        model: Modelo entrenado (perceptrón o red neuronal)
        X: Datos de entrada
        y: Etiquetas
        title: Título del gráfico
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title(title)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.colorbar()
    plt.show()


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO 1: PERCEPTRÓN - Clasificación Linealmente Separable")
    print("=" * 70)
    
    # Generar datos linealmente separables
    X_train, y_train = generate_linearly_separable_data(n_samples=100)
    
    # Entrenar perceptrón
    perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron.fit(X_train, y_train)
    
    # Evaluar
    accuracy = perceptron.score(X_train, y_train)
    print(f"\nPrecisión del perceptrón: {accuracy:.2%}")
    print(f"Pesos finales: {perceptron.weights}")
    print(f"Sesgo final: {perceptron.bias}")
    
    print("\n" + "=" * 70)
    print("EJEMPLO 2: RED NEURONAL DE UNA CAPA - Clasificación Multiclase")
    print("=" * 70)
    
    # Generar datos para clasificación multiclase
    from sklearn.datasets import make_classification
    X_multi, y_multi = make_classification(n_samples=200, n_features=4, 
                                          n_classes=3, n_informative=4, 
                                          n_redundant=0, random_state=42)
    
    # Entrenar red de una capa
    slnn = SingleLayerNeuralNetwork(learning_rate=0.1, n_iterations=500, n_neurons=3)
    slnn.fit(X_multi, y_multi)
    
    # Evaluar
    accuracy = slnn.score(X_multi, y_multi)
    print(f"\nPrecisión de la red de una capa: {accuracy:.2%}")
    
    print("\n" + "=" * 70)
    print("EJEMPLO 3: RED NEURONAL MULTICAPA - Problema XOR")
    print("=" * 70)
    
    # Generar datos XOR (no linealmente separable)
    X_xor, y_xor = generate_xor_data(n_samples=100)
    
    # Entrenar red multicapa
    mlp = MultiLayerNeuralNetwork(learning_rate=0.1, n_iterations=1000,
                                  hidden_layers=[5, 5], activation='tanh')
    mlp.fit(X_xor, y_xor)
    
    # Evaluar
    accuracy = mlp.score(X_xor, y_xor)
    print(f"\nPrecisión de la red multicapa: {accuracy:.2%}")
    
    print("\n" + "=" * 70)
    print("¡Implementación completada exitosamente!")
    print("=" * 70)

