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
epochs=50000
learning_rate=0.001
# Train the neural network
loss_record = network.train(x_train, y_train, epochs, learning_rate, y_train_mean, y_train_std)

output_train = network.forward(x_train)*y_train_std+y_train_mean
y_train_original = y_train*y_train_std+y_train_mean

output_test = network.forward(x_test)*y_test_std+y_test_mean
y_test_original = y_test*y_test_std+y_test_mean

test_ERMS = np.sqrt((np.sum(np.square(y_test_original-output_test)))/192)

print(f"training ERMS: {loss_record[-1]/596} \ntest ERMS: {test_ERMS}")


numbers = range(1,577)
plt.plot(numbers, y_train_original, color = 'blue', label = 'y_train')
plt.plot(numbers, output_train, color = 'red', label = 'predict')
plt.xlabel('case')
plt.ylabel('Heating Load')
plt.title('Train Predict') 
plt.legend()
plt.show()

numbers = range(1,193)
plt.plot(numbers, y_test_original, color = 'blue', label = 'y_test')
plt.plot(numbers, output_test, color = 'red', label = 'predict')
plt.xlabel('case')
plt.ylabel('Heating Load')
plt.title('Test Predict') 
plt.legend()
plt.show()

numbers = range(1,epochs+1)
plt.plot(numbers, loss_record, color = 'blue', label = 'loss')
plt.xlabel('epoches')
plt.ylabel('E')
plt.title('Loss') 
plt.legend()
plt.show()