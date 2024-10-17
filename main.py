import readfile as rd
import neuralnetwork as nn
import matplotlib.pyplot as plt

file = '2024_energy_efficiency_data.csv'
train_ratio = 0.75

x_train,y_train,x_test,y_test,y_train_mean,y_train_std,y_test_mean,y_test_std = rd.read_file(file,train_ratio)

# Define the network
input_size = 17
hidden_size = [16,8,8]
output_size = 1

# Create the neural network
network = nn.DeepNeuralNetwork(input_size, hidden_size, output_size)
epochs=10000
learning_rate=0.001
# Train the neural network
loss_record = network.train(x_train, y_train, epochs, learning_rate, y_train_mean, y_train_std)

# Test the network
# print(x_test[0].reshape(1,17)*x_test_std+x_test_mean)
output = network.forward(x_test)*y_test_std+y_test_mean
y = y_test*y_test_std+y_test_mean

numbers = range(1,193)
plt.plot(numbers, output, color = 'red', label = 'predict')
plt.plot(numbers, y, color = 'blue', label = 'y_test')
plt.xlabel('case')
plt.ylabel('Heating Load')
plt.title('Predict') 
plt.legend()
plt.show()

numbers = range(1,epochs+1)
plt.plot(numbers, loss_record, color = 'blue', label = 'loss')
plt.xlabel('epoches')
plt.ylabel('ERM')
plt.title('Loss') 
plt.legend()
plt.show()