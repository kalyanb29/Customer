import pandas as pd
import os


data = pd.read_csv(os.getcwd() + '/train/all_features.csv', sep=' ')
label_rep = data['repeattrips']
features = data.
index = list(data)
print(index)