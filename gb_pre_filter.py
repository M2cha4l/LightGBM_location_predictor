import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
import math
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def foldgbm(X_f, y_f):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # K折交叉验证
    # model = model_f
    X = X_f
    y = y_f
    i = 1
    auc_f, mae, rmse, r2, result = [], [], [], [], []
    for train_index, test_index in kf.split(X, y):
        #         print('\n{} of kfold {}'.format(i,kf.n_splits))
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        gbm = lgb.train(params, lgb_train)
        pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_test, pred_test)
        pred_test = pred_test.round(0)
        pred_test = pred_test.astype(int)
        auc_f.append(auc)
        # mae.append(mean_absolute_error(y_test, pred_test))
        # rmse.append(math.sqrt(mean_absolute_error(y_test, pred_test)))
        # r2.append(r2_score(y_test, pred_test))
    result.append(sum(auc_f) / len(auc_f))
    # result.append(sum(mae) / len(mae))
    # result.append(sum(rmse) / len(rmse))
    # result.append(sum(r2) / len(r2))
    return result
aut_list = []
for i in range(100):
    print(f'目前是滴{i}轮')
    df = pd.read_csv('dataset_after_missing_values.csv').drop(['uid'],axis=1)
    dfl = df
    df['feature'] = df['feature'].apply(lambda x:str(x))
    data_1 = df[df["feature"].str.contains("1")]
    data_2 = df[df["feature"].str.contains("0")]
    data_2 = data_2.sample(frac=1).reset_index(drop=True).loc[1:387,]
    df = pd.concat([data_1,data_2],axis=0)
    df['feature'] = df['feature'].apply(lambda x:int(x))
    df = df.sample(frac=1).reset_index()
    percent =int(0.8*df.shape[0])
    X = df.drop(['feature','index'],axis=1)
    y = df['feature']
    X_test = df.drop(['feature','index'],axis=1).iloc[percent:,]
    y_test = df.loc[percent:,'feature']
    X_train = df.drop(['feature','index'],axis=1).loc[:percent,]
    y_train= df.loc[:percent,'feature']
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
                            valid_sets=lgb_eval,
                            )
    k_f = foldgbm(X, y)
    aut_list.append(k_f)
    print('lgb结果', k_f)
    X_all = dfl.drop(['feature'],axis=1)
    y_all= dfl['feature']
    lgb_pri = lgb.Dataset(X_all, y_all)
    ypred = gbm.predict(X_all, num_iteration=gbm.best_iteration)
    # ypred=ypred.round(0)
    # ypred=ypred.astype(int)
    print(type(ypred))
    ypred = pd.DataFrame(ypred)
    # print(roc_auc_score(ypred,y_test))
    ypred.to_csv(f'./dataset1/{i}.csv')
sum_f = pd.read_csv('./dataset1/1.csv')
for i in range(2,100):

    da = pd.read_csv(f'./dataset1/{i}.csv')
    for k in range(1, sum_f.shape[0]):
        sum_f.iloc[k, 1] += da.iloc[k, 1]
sum_f.to_csv('./dataset1/sum.csv')
print('共有',len(aut_list),'个模型')
aut_list = pd.DataFrame(aut_list).to_csv('./dataset1/auc.csv')

