import readfile as rd
import neuralnetwork as nn
import numpy as np 

file = '2024_energy_efficiency_data.csv'
train_ratio = 0.75

x_train,y_train,x_test,y_test = rd.read_file(file,train_ratio)
print(f"x_train :\n {x_train.shape}")
print(f"y_train :\n {y_train.shape}")
print(f"x_test :\n {x_test.shape}")
print(f"y_test :\n {y_test.shape}")
# layers = [x.shape[1], 64, 32, 1]
# learning_rate=0.01
# epochs=100
# batch_size=32

# trained_parameters = nn.mini_batch_SGD(x, y, layers, learning_rate, epochs, batch_size)

# Define the network
input_size = 17
hidden_size = [128, 256, 64]
output_size = 1

# Create the neural network
network = nn.DeepNeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
network.train(x_train, y_train, epochs=1000, learning_rate=0.01)

# Test the network
print(x_test[0].reshape(1,17))
output = network.forward(x_test[0].reshape(1,17))
print(network.weights)
print("Predicted Output:")
print(output)
