import readfile as rd
import numpy as np
file = '2024_energy_efficiency_data.csv'
train_ratio = 0.75

x_train,y_train,x_test,y_test = rd.read_file(file,train_ratio)
print(f"x_train :\n {x_train.shape}")
print(f"y_train :\n {y_train.shape}")
print(f"x_test :\n {x_test.shape}")
print(f"y_test :\n {y_test.shape}")
print(x_test[0].shape)

a = np.array([1,2,3,4,5,6,7,8,9,10])

print(a.shape)