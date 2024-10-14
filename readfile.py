import pandas as pd
import numpy as np



def one_hot_encode(data,name):
    
    max_val = int(max(data))
    min_val = int(min(data))
    length = max_val - min_val
    one_hot_matrix = np.zeros((len(data.axes[0]), length+1), dtype=float)
    
    for i, value in enumerate(data):
        one_hot_matrix[i, int(value) - min_val] = 1
    
    encode = pd.DataFrame(one_hot_matrix)
    columns_name = []
    for i in range(min_val ,max_val+1):
        columns_name.append(name+"_"+str(i))
    encode.columns = columns_name

    return encode
    

def read_file(file_path, train_ratio):
    np.random.seed()
    data = pd.read_csv(file_path)

    #Converting into a Pandas dataframe
    df = pd.DataFrame(data)
    #Print the dataframe:
    # print(f"Employee data : \n{df}")

    #Extract categorical columns from the dataframe
    #Here we extract the columns with object datatype as they are the categorical columns
    categorical_columns = ['Orientation','Glazing Area Distribution']
    
    for i in categorical_columns:
        encode = one_hot_encode(df[i],i)
        df = pd.concat([df, encode], axis=1)
    df_encoded = df.drop(categorical_columns, axis=1)
    # Drop the original categorical columns

    # Display the resulting dataframe
    # print(f"Encoded Employee data : \n{df_encoded}")
    x = np.array(df_encoded.drop('Heating Load', axis=1))
    y = np.array(df_encoded['Heating Load'])
    permutation = np.random.permutation(len(y))
    train_size = int(train_ratio * len(y))
    train_indices = permutation[:train_size]
    test_indices = permutation[train_size:]
    x_train = x[train_indices]
    y_train = y[train_indices].reshape(train_size,1)
    x_test = x[test_indices]  
    y_test = y[test_indices].reshape(len(y)-train_size,1)
    
    return x_train,y_train,x_test,y_test

