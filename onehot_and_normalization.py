import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('dataset_after_missing_values.csv')
df = pd.get_dummies(df,columns=['tudi'])
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
df=df.apply(max_min_scaler)
df.to_csv('final_dataset.csv')