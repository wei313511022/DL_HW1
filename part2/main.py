import readfile as rd
import neuralnetwork as nn
import matplotlib.pyplot as plt
import numpy as np

file = '2024_ionosphere_data.csv'
train_ratio = 0.8

x_train_standardized,y_train,x_test_standardized,y_test = rd.read_file(file,train_ratio)

input_size = 33  # Number of input features
hidden_layers = [64, 4]  # Two hidden layers
output_size = 2  # Two classes for binary classification
epochs = 50
learning_rate = 0.01
batch_size = 280

# Create random data for training (X: features, y: one-hot encoded labels)

x_train_standardized = x_train_standardized.astype(np.float64)
y_train = y_train.astype(np.float64)

# Initialize the neural network
network = nn.SimpleNeuralNetwork(input_size, hidden_layers, output_size)

# Train the network
loss_record = network.train(x_train_standardized, y_train, epochs, learning_rate, batch_size)

numbers = range(1,epochs+1)
plt.plot(numbers, loss_record, color = 'blue', label = 'loss')
plt.xlabel('epoches')
plt.ylabel('cross-entropy')
plt.title('Loss') 
plt.legend()
plt.show()

y_predict,z_final = network.forward(x_test_standardized)
y_predict = (y_predict >= 0.5).astype(float)
j = 0
error = 0

for i in range(71):
    if y_test[j][0] != y_predict[j][0]:
        # print(f"{y_test[j][0]}   {y_predict[j][0]}")
        error += 1
    j +=1
print(f"test error rate: {error/71}")



y_predict,z_final = network.forward(x_train_standardized)

red_points = []
blue_points = []
for i in range(280):
    if y_train[i][0] == 1:
        red_points.append((z_final.T[0][i],z_final.T[1][i]))
    else:
        blue_points.append((z_final.T[0][i],z_final.T[1][i]))
red_points = np.array(red_points)
blue_points = np.array(blue_points)
plt.scatter(red_points[:, 0], red_points[:, 1], c='red', label='Class 1')
plt.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', label='Class 0')
plt.title(f"nodes: {hidden_layers[1]} epoch: {epochs}") 
plt.legend()
plt.show()

y_predict = (y_predict >= 0.5).astype(float)
j = 0
error = 0

for i in range(280):
    # print(f"{y_train[j][0]}   {y_predict[j][0]}")
    if y_train[j][0] != y_predict[j][0]:
        
        error += 1
    j +=1
print(f"y_train error rate: {error/280}")
