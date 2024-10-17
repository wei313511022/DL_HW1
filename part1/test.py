import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("2024_energy_efficiency_data.csv")

df = pd.DataFrame(data)


plt.scatter(df['Wall Area'], df['Heating Load'], color = 'red')

plt.xlabel('Wall Area')
plt.ylabel('Heating Load')
plt.legend()
plt.show()