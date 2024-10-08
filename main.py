import readfile as rd
import neuralnetwork as nn
import numpy as np 

x,y = rd.read_file('2024_energy_efficiency_data.csv')

layers = [x.shape[1], 64, 32, 1]
learning_rate=0.01
epochs=100
batch_size=32

trained_parameters = nn.mini_batch_SGD(x, y, layers, learning_rate, epochs, batch_size)
