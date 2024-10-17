import pandas as pd
import numpy as np

def read_file(file,train_ratio):
    np.random.seed()
    data = pd.read_csv(file, header = None)
    df = pd.DataFrame(data)
    y = np.zeros((351,2))
    j = 0
    for i in df[34]:
        if i=='g':
            y[j][0] = 1
        else:
            y[j][1] = 1
        j+=1

    x = np.array(df.drop([1,34], axis=1))
    
    permutation = np.random.permutation(len(y))
    train_size = int(train_ratio * len(y))
    train_indices = permutation[:train_size]
    test_indices = permutation[train_size:]

    x_train = x[train_indices]
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train_standardized = (x_train - x_train_mean) / x_train_std

    x_test = x[test_indices]  
    x_test_mean = np.mean(x_test, axis=0)
    x_test_std = np.std(x_test, axis=0)
    x_test_standardized = (x_test - x_test_mean) / x_test_std

    y_train = y[train_indices]
    y_test = y[test_indices]

    return x_train_standardized,y_train,x_test_standardized,y_test
    