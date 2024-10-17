import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Layers configuration
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        
        # He Initialization for weights
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.layer_inputs = []
        self.layer_activations = []
        
        # Forward pass through hidden layers
        activation = X
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            activation = self.relu(z)
            self.layer_activations.append(activation)
        
        # Forward pass through output layer with softmax
        z_final = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(z_final)
        output = self.softmax(z_final)
        self.layer_activations.append(output)
        
        return output,z_final

    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # To prevent log(0) error
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        return loss
    
    def backward(self, X, y_true, y_pred, learning_rate):
        # Derivative of cross-entropy loss with softmax output
        delta = (y_pred - y_true) / y_true.shape[0]
        
        # Backpropagation loop through layers
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_activations[i - 1].T, delta) if i > 0 else np.dot(X.T, delta)
            dB = np.sum(delta, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
            
            if i > 0:
                # Calculate delta for the next layer back
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.layer_inputs[i - 1])
    
    def train(self, X, y, epochs, learning_rate, batch_size):
        m = X.shape[0]
        loss_record = []
        
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                # Forward pass
                y_pred,numb = self.forward(X_batch)
            
                # Backward pass
                self.backward(X_batch, y_batch, y_pred, learning_rate)
            
            y_pred,numb = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred)
            loss_record.append(loss)
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch} - Loss: {loss:.4f}')
        return loss_record