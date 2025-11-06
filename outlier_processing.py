import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.figure(figsize=(9,9)) # 画布
df = pd.read_excel('dataset.xlsx')
df1 = df[0:1000]
plt.grid(True) # 显示网格
means_opt = ['xiaoqu_dict','gongsi_dist']
figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 9), dpi=80)
k=0
for i in range(2):
    axis[i].boxplot(df1[means_opt[k]],
                       sym="g+",  # 异常点形状，默认为蓝色的“+”
                       showmeans=True  # 是否显示均值，默认不显示
                       )
    axis[i].grid(linestyle="--", alpha=0.5)
    axis[i].set_title(means_opt[k])
    k+=1
plt.suptitle("各变量箱线图")
plt.savefig(f"./figure/subbox.png")


