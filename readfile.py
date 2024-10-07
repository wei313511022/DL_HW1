import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np 

def read_file(file_path):
    data = pd.read_csv(file_path)

    #Converting into a Pandas dataframe
    df = pd.DataFrame(data)
    #Print the dataframe:
    # print(f"Employee data : \n{df}")

    #Extract categorical columns from the dataframe
    #Here we extract the columns with object datatype as they are the categorical columns
    categorical_columns = ['Orientation','Glazing Area Distribution']
    
    #Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([df, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    # Display the resulting dataframe
    # print(f"Encoded Employee data : \n{df_encoded}")
    temp = df_encoded.drop('Heating Load', axis=1)
    x = np.array(temp).flatten(order='F')
    y = df_encoded['Heating Load']
    
    return x,y

