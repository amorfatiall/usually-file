#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Time    :2019/4/29 17:22
# Author  :FanQin
# File    :titanic
# Software: PyCharm

# 数据预处理
import pandas as pd
import numpy as np
titanic = pd.read_csv('titanic_train.csv')
print(titanic.describe())
titanic = titanic.drop_duplicates()
print(titanic.describe())
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print(titanic.describe())
# 对Sex列进行符号替代
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
# 对于Embarked列进行填充和符号替代
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
# titanic.to_csv('TT.csv')
# 使用回归算法进行预测
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# 选择要使用的特征
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# 导入线性回归
alg = LinearRegression()
# 进行交叉验证的选择，三倍交叉验证
kf = KFold(n_splits=3, shuffle=False, random_state=1)
# n_splits/n_folds都发生错误 https://blog.csdn.net/weixin_40283816/article/details/83242777
predictions = []
for train, test, in kf.split(titanic):
    # 将训练数据拿出来，对原始数据取到建立好的特征 然后取出用于训练的那一部分
    train_predictors = (titanic[predictors].iloc[train, :])
    # print(train_predictors)

    # 获取到数据集中交叉分类好的标签，即是否活了下来
    train_target = titanic["Survived"].iloc[train]
    # print(train_target)

    # 将数据放进去做训练， .fit 表示把选择的算法应用在当前的数据上
    alg.fit(train_predictors, train_target)
    # 训练完后，使用测试集进行测试误差  alg.predict() 对测试集数据进行预测
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    # 把测试结果导入到predictors里面
    predictions.append(test_predictions)

#使用线性回归得到的结果是在区间[0,1]上的某个值，需要将该值转换成0或1
predictions = np.concatenate(predictions, axis=0)
#print(predictions)
#print(type(titanic["Survived"]))

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
predictions.dtype = "float64"

titanic["Survived"] = titanic["Survived"].astype(float)
print("测试数据总数量", len(predictions))
#print("正确的数量：", sum(predictions[predictions == titanic["Survived"]])) python2的语法
print("正确的数量：", sum(predictions == titanic["Survived"]))  #Python3的语法
accuracy = sum(predictions == titanic["Survived"]) / len(predictions)
print("准确率为：", accuracy)

# 线性回归的准确率为： 0.7833894500561167，对于二分类来说其实是比较低的，所以需要换一个方法
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
blg = LogisticRegression(random_state=1)
scores = cross_val_score(blg, titanic[predictors], titanic['Survived'], cv=3)
print(scores.mean())
# 逻辑回归的准确率为：0.7878787878787877
# https://blog.csdn.net/qq_36523839/article/details/80707678?tdsourcetag=s_pcqq_aiomsg 交叉验证的解释和例子，解释了cv
titanic_test = pd.read_csv('titanic_test.csv')
titanic_test = titanic_test.drop_duplicates()
# 测试集数据清洗处理
titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
# 对Sex列进行符号替代
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
# 对于Embarked列进行填充和符号替代
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
predictors_1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# 预测并产生提交文件
clg = LogisticRegression(random_state=1)
clg.fit(titanic[predictors], titanic["Survived"])
predictions = clg.predict(titanic_test[predictors_1])
# Create a new dataframe with only the columns Kaggle wants from the data set
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic_result_logistic.csv')

# 使用随机森林改进模型
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = KFold(n_splits=3, shuffle=False, random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())
# 随机森林准确率为0.7856341189674523
# 随机森林参数调整，模型调优
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
kf = KFold(n_splits=3, shuffle=False, random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())
# 调参后的随机森林准确率为0.8159371492704826

# 新建一列特征：家庭人口数
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
# 新建一列特征：姓名长度
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

import re
# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title. Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v
print(pd.value_counts(titles))
titanic["Title"] = titles

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)


from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(n_splits=3, shuffle=False, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)











