# 在环境变量中加入安装的Graphviz路径

from sklearn.datasets import load_iris
import lightgbm as lgb

iris = load_iris()
lgb_clf = lgb.LGBMClassifier()
lgb_clf.fit(iris.data, iris.target)
lgb.create_tree_digraph(lgb_clf, tree_index=1)
