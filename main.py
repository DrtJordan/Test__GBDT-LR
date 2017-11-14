#coding=utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn import  metrics
from sklearn.model_selection import train_test_split

# 导入数据
# X = pd.read_table('vecs_new.txt', header=None, sep=',')
# y = pd.read_table('labels_new.txt', header=None)

iris = load_iris()
X, y = iris.data, iris.target

y = [1 if ele > 1 else 0  for ele in y]

# 切分为测试集和训练集，比例0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

# 弱分类器的数目
n_estimator = 10
# 调用GBDT分类模型
grd = GradientBoostingClassifier(n_estimators=n_estimator)

# 调用one-hot编码。
grd_enc = OneHotEncoder()

# 调用LR分类模型。
grd_lm = LogisticRegression()

# 使用X_train训练GBDT模型，后面用此模型构造特征
grd.fit(X_train, y_train)

# 直接进行预测，查看AUC得分
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = metrics.roc_curve(y_test, y_pred_grd)
roc_auc = metrics.auc(fpr_grd, tpr_grd)
print
'predict', roc_auc

# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])

# 使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

# 用训练好的LR模型多X_test做预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]

# 根据预测结果输出
fpr_grd_lm, tpr_grd_lm, _ = metrics.roc_curve(y_test, y_pred_grd_lm)
roc_auc = metrics.auc(fpr_grd_lm, tpr_grd_lm)
print
'predict', roc_auc

print("AUC Score :", (metrics.roc_auc_score(y_test, y_pred_grd_lm)))