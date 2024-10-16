import numpy as np

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the neural network class with more layers
class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize weights and biases for multiple layers
        self.weights = []
        self.biases = []
        
        # Create weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]) )
            self.biases.append(np.random.rand(layer_sizes[i + 1]) )

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = []
        
        # Forward propagation through each layer
        current_output = X
        for i in range(len(self.weights) - 1):
            layer_input = np.dot(current_output, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            current_output = relu(layer_input)
            self.layer_outputs.append(current_output)
        
        # Last layer (output layer)
        final_input = np.dot(current_output, self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(final_input)
        final_output = relu(final_input)  # Use ReLU for simplicity; can also use other activation
        self.layer_outputs.append(final_output)
        
        return final_output

    def backward(self, X, y, output, learning_rate):
        # Error at the output
        error= np.mean(y - output)
        
        print(f"error size {error.shape}")
        
        # Backpropagate the error
        d_output = error * relu_derivative(output)
        deltas = [d_output]
        
        # Backpropagate through each layer
        for i in reversed(range(len(self.weights) - 1)):
            error_hidden = deltas[-1].dot(self.weights[i + 1].T)
            d_hidden = error_hidden * relu_derivative(self.layer_outputs[i])
            deltas.append(d_hidden)
        
        # Reverse deltas to match the forward pass order
        deltas.reverse()
        
        # Update weights and biases
        current_input = X
        for i in range(len(self.weights)):
            self.weights[i] += current_input.T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0) * learning_rate
            current_input = self.layer_outputs[i]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass and weight update
            self.backward(X, y, output, learning_rate)
            
            # Print loss for every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.sqrt(np.mean(np.square(y - output)))
                print(f"Epoch {epoch}, Loss: {loss}")

# # Example Usage:
# # Replace with your actual data (576 samples with 17 features)
# X = np.random.rand(576, 17)
# y = np.random.randint(2, size=(576, 1))

# # Define the network dimensions
# input_size = 17
# hidden_sizes = [32, 16, 8]  # More hidden layers with different sizes
# output_size = 1

# # Create the neural network
# nn = DeepNeuralNetwork(input_size, hidden_sizes, output_size)

# # Train the neural network
# nn.train(X, y, epochs=10000, learning_rate=0.01)

# # Test the network
# output = nn.forward(X)
# print("Predicted Output:")
# print(output)
