import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
df = pd.read_excel('dataset.xlsx')
df1 = df[0:1000]
plt.style.use('fivethirtyeight')
figure, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), dpi=80)
# 进行数据的可视化（证明是均匀分布的）
means_opt = ['gongsi_poi','gouwu_poi','gonggong_poi','fengjing_poi','canyin_poi','tiyu_poi','shenghuo_poi','shangwu_poi','kejiao_poi','jinrong_poi','jiaotong_poi','zhusu_poi','zhengfu_poi','yiliao_poi','gaocheng_mean','podu_mean','gdp_yuan','xiaoqu_dict','gongsi_dist']
k=0
# for i in range(3):
#     for i1 in range(2):
#         axis[i,i1].scatter(df1['uid'],df1[means_opt[k]],s=50)
#         axis[i,i1].grid(linestyle="--", alpha=0.5)
#         axis[i,i1].set_title(means_opt[k])
#         axis[i,i1].set_xticks([])
#         k+=1
# plt.suptitle("特征散点图（部分）")
# plt.savefig(f"./figure/subfigure.png")
# 缺失值处理POI
for column in means_opt:
    print(df[column].mean())
    df[column].fillna(df[column].mean(),inplace=True)
    print(column,'填充完毕')
# 众数处理
df['tudi'].fillna(df['tudi'].mode()[0],inplace=True)
df.to_csv('dataset_after_missing_values.csv')
