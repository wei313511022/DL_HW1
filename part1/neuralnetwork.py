import numpy as np

# sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
            #print(current_output)
            layer_input = np.dot(current_output, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            current_output = sigmoid(layer_input)
            self.layer_outputs.append(current_output)
        
        # Last layer (output layer)
        final_input = np.dot(current_output, self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(final_input)
        final_output = final_input  # Use sigmoid for simplicity; can also use other activation
        self.layer_outputs.append(final_output)
        
        return final_output

    def backward(self, x, y, output, learning_rate):
        # Error at the output
        
        
        # Derivative of the loss with respect to the output
        d_loss_output = -2 * (y - output)
        # print(f"d_loss_output  {d_loss_output}")
        
        # Backpropagation
        d_output = d_loss_output * sigmoid_derivative(self.layer_inputs[-1])
        # print(f"d_output  {d_output}")
        d_weights = []
        d_biases = []
        
        # Loop backward through layers
        for i in reversed(range(len(self.weights))):
            # Compute weight and bias gradients
            input_to_layer = x if i == 0 else self.layer_outputs[i - 1]
            # print(f"input_to_layer.shape  {i}  {input_to_layer}")
            d_weight = np.dot(input_to_layer.T, d_output)
            # print(f"d_weight.shape  {i}  {d_weight}")
            d_bias = np.sum(d_output, axis=0, keepdims=True)
            
            # Store the gradients
            d_weights.insert(0, d_weight)
            d_biases.insert(0, d_bias)
            
            # Propagate the error backwards (except for the input layer)
            if i > 0:
                d_output = np.dot(d_output, self.weights[i].T) * sigmoid_derivative(self.layer_inputs[i - 1])
        
        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            a = np.squeeze(d_biases[i])
            self.biases[i] -= learning_rate * a
        
    def train(self, X, y, epochs, learning_rate, batch_size, y_train_mean, y_train_std):
        m = X.shape[0]  # Number of samples
        loss_record = []
        
        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, m, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass and weight update
                self.backward(X_batch, y_batch, output, learning_rate)
                
            # Compute loss for the entire dataset after each epoch
            output_full = self.forward(X)
            loss = np.sum(np.square((y - output_full) * y_train_std))
            loss_record.append(loss)
            
            # Print loss for every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return loss_record
