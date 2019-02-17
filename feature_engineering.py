#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'None'
__author__ = 'None'
__mtime__ = 'None'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛

┌───┐   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌
│Esc│   │ F1│ F2│ F3│ F4│ │ F5│ F6│ F7│ F8│ │ F9│F10│F11│F12│ │P/S│S L│P/B│  ┌┐    ┌┐    ┌┐  │
└───┘   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───────┐
│~ `│! 1│@ 2│# 3│$ 4│% 5│^ 6│& 7│* 8│( 9│) 0│_ -│+ =│ BacSp │ │Ins│Hom│PUp│ │N L│ / │ * │ - │   │
├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─────┤ 
│ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{ [│} ]│ | \ │ │Del│End│PDn│ │ 7 │ 8 │ 9 │   │   │
├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤ 
│ Caps │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  │                   │ 4  │ 5 │ 6 │   │   │
├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────────┤  
│ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│  Shift   │     │ ↑ │       │ 1  │ 2 │ 3 │   │   │
├─────┬──┴─┬─┴──┬┴───┴───┴───┴───┴───┴──┬┴───┼───┴┬────┬────┤ 
│ Ctrl│    │Alt │         Space         │ Alt│    │    │Ctrl│ │ ← │ ↓ │ → │ │   0   │ . │←─┘│    │
└─────┴────┴────┴───────────────────────┴────┴────┴────┴────┘ 
"""

import pandas as pd
from sklearn import linear_model

data_train = pd.read_csv('DATASET/train.csv')
# 数据预处理一 : 补充缺失值
# 缺失：Age  Cabin  Embarked
data_train['Age'].fillna(int(data_train['Age'].mean()), inplace=True)
data_train.loc[(data_train['Cabin'].notnull()), 'Cabin'] = 'YES'
data_train.loc[(data_train['Cabin'].isnull()), 'Cabin'] = 'NO'
data_train['Embarked'].fillna('S',inplace=True)

# 数据预处理二 : 数据one-hot处理(哑变量处理类别数据)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Sex, dummies_Embarked, dummies_Pclass], axis=1)
df.drop(['Cabin', 'Sex', 'Embarked', 'Pclass'], axis=1, inplace=True)

# 数据预处理三 : 数据数据标准差处理
a = df.Age
b = df.Fare
df['Age_scaled'] = (a-a.mean())/(a.std(ddof=0))
df['Fare_scaled'] = (b-b.mean())/(b.std(ddof=0))
df = df.drop('Age', axis=1)
df = df.drop('Fare', axis=1)

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
x = train_np[:, 1:]

# penalty:制定正则化策略
# C:一个浮点数，它指定了惩罚系数的倒数。如果它的值越小，则正则化越大
# tol：一个浮点数，指定判断迭代收敛与否的一个阈值
# verbose：一个正数。用于开启/关闭迭代中间输出的日志。
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, verbose=True)
# clf.fit(x, y)

data_test = pd.read_csv('DATASET/test.csv')

# 测试数据预处理
data_test['Age'].fillna(int(data_test['Age'].mean()), inplace=True)
data_test.loc[(data_test['Cabin'].notnull()), 'Cabin'] = 'YES'
data_test.loc[(data_test['Cabin'].isnull()), 'Cabin'] = 'NO'

print(data_test.info())








'''
训练集数据信息
RangeIndex: 891 entries, 0 to 890 
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
'''