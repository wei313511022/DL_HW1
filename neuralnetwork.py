import numpy as np
import csv

np.random.seed(42)
# The rest of the neural network code remains the same
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initialize_parameters(layer_sizes):
    parameters = {}
    L = len(layer_sizes)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
    
    return parameters

def forward_propagation(X, parameters):
    L = len(parameters) // 2
    caches = {'A0': X}
    
    A = X
    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        Z = np.dot(W, A) + b
        A = sigmoid(Z)
        
        caches['Z' + str(l)] = Z
        caches['A' + str(l)] = A
    
    return A, caches

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    loss = -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return np.squeeze(loss)

def backward_propagation(Y, parameters, caches):
    gradients = {}
    L = len(parameters) // 2
    m = Y.shape[1]
    
    A_final = caches['A' + str(L)]
    dA = - (np.divide(Y, A_final) - np.divide(1 - Y, 1 - A_final))
    
    for l in reversed(range(1, L + 1)):
        dZ = dA * sigmoid_derivative(caches['Z' + str(l)])
        dW = 1/m * np.dot(dZ, caches['A' + str(l-1)].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W' + str(l)].T, dZ)
        
        gradients['dW' + str(l)] = dW
        gradients['db' + str(l)] = db
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]
    
    return parameters

def mini_batch_SGD(X, Y, layers, learning_rate, epochs, batch_size):
    parameters = initialize_parameters(layers)
    m = X.shape[0]  # number of training examples
    
    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation, :]
        Y_shuffled = Y[permutation, :]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size, :]
            Y_batch = Y_shuffled[i:i+batch_size, :]
            
            # Forward propagation
            Y_hat, caches = forward_propagation(X_batch, parameters)
            
            # Compute the loss (optional to print during training)
            loss = compute_loss(Y_batch, Y_hat)
            
            # Backward propagation
            gradients = backward_propagation(Y_batch, parameters, caches)
            
            # Update parameters
            parameters = update_parameters(parameters, gradients, learning_rate)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return parameters

# Example usage



# Define the architecture of the neural network
# layers = [X.shape[0], 64, 32, 1]  # Input -> 64 hidden -> 32 hidden -> Output
# trained_parameters = mini_batch_SGD(X, Y, layers, learning_rate=0.01, epochs=100, batch_size=32)
