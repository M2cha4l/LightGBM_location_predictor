from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
target_class = 1
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
df = pd.read_csv('dataset_after_missing_values.csv').drop(['uid','jingdu','weidu'], axis=1)
df['feature'] = df['feature'].apply(lambda x: str(x))
data_1 = df[df["feature"].str.contains("1")]
data_2 = df[df["feature"].str.contains("0")]
data_2 = data_2.sample(frac=1).reset_index(drop=True).loc[1:387, ]
df = pd.concat([data_1, data_2], axis=0)
df['feature'] = df['feature'].apply(lambda x: int(x))
df = df.sample(frac=1).reset_index()
percent = int(0.8 * df.shape[0])
X = df.drop(['feature', 'index'], axis=1)
y = df['feature']
X_test = df.drop(['feature', 'index'], axis=1).iloc[percent:, ]
y_test = df.loc[percent:, 'feature']
X_train = df.drop(['feature', 'index'], axis=1).loc[:percent, ]
y_train = df.loc[:percent, 'feature']
# 这个是取不变的测试集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc', 'binary'},
    'num_leaves': 30,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
gbm = lgb.train(params, lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_eval],
                )
# fig2 = plt.figure(figsize=(20, 20))
# ax = fig2.subplots()
# lgb.plot_tree(gbm, tree_index=1, ax=ax)
# plt.show()

for i in range(len(gbm.feature_name())):
    print(list(gbm.feature_importance())[i],gbm.feature_name()[i])


