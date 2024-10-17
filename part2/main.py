import readfile as rd
import neuralnetwork as nn
import matplotlib.pyplot as plt
import numpy as np

file = '2024_ionosphere_data.csv'
train_ratio = 0.8

x_train_standardized,y_train,x_test_standardized,y_test = rd.read_file(file,train_ratio)

input_size = 33  # Number of input features
hidden_layers = [64, 32]  # Two hidden layers
output_size = 2  # Two classes for binary classification

# Create random data for training (X: features, y: one-hot encoded labels)

x_train_standardized = x_train_standardized.astype(np.float64)
y_train = y_train.astype(np.float64)

# Initialize the neural network
network = nn.SimpleNeuralNetwork(input_size, hidden_layers, output_size)

# Train the network
network.train(x_train_standardized, y_train, epochs=1000, learning_rate=0.01)

y_predict = np.array(network.forward(x_test_standardized))
y_predict = (y_predict >= 0.5).astype(float)
j = 0
error = 0

for i in range(71):
    
    if y_test[j][0] != y_predict[j][0]:
        print(f"{y_test[j][0]}   {y_predict[j][0]}")
        error += 1
    j +=1
print(f"error: {error/71}")
