import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import math
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入数据
df = pd.read_csv('dataset_after_missing_values.csv').drop(['uid'], axis=1)
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
                valid_sets=lgb_eval,
                )
ypred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

fpr, tpr, threshold = roc_curve(y_test, ypred)
roc_auc = roc_auc_score(y_test,ypred)   # 准确率代表所有正确的占所有数据的比值
print('roc_auc:', roc_auc)
lw = 2
# plt.subplot(1,1,1)
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('1 - specificity')
# plt.ylabel('Sensitivity')
# plt.title('ROC', y=0.5)
# plt.legend(loc="lower right")
# plt.savefig(f"./figure/roc.png")
# 混淆矩阵绘制
def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            # annotate主要在图形中添加注释
            # 第一个参数添加注释
            # 第二个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()
ypred=ypred.round(0)
ypred=ypred.astype(int)
cm_plot(y_test,ypred,'matrix')

