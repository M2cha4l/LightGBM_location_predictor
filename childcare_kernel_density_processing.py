import pandas as pd
import numpy as np
df = pd.read_csv('childcare_kernel_density.csv')
# df = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
df=df.apply(max_min_scaler)
df.to_csv('m_tuoyu.csv')