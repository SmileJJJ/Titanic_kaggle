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
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

data_train = pd.read_csv('DATASET/train.csv')

# data_train.info()

fig1 = plt.figure(1)
fig1.set(alpha=0.2)
plt.subplot2grid((3,2),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')     # 柱状图
plt.title(u"获救情况")
plt.ylabel(u"人数")

plt.subplot2grid((3,2),(0,1))             # 在一张大图里分列几个小图
data_train.Pclass.value_counts().plot(kind='bar')     # 柱状图
plt.title(u"乘客等级情况")
plt.ylabel(u"人数")

plt.subplot2grid((3,2),(2,1))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((3,1),(1,0))
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.ylabel(u'密度')
plt.title(u'各年龄段乘客等级情况')
plt.legend(('1级乘客', '2级乘客', '3级乘客'), loc='best')

plt.subplot2grid((3,2),(2,0))
data_train.Embarked.value_counts().plot(kind='bar')
plt.ylabel(u'人数')
plt.xlabel(u'港口')
plt.title(u'登船港口情况')

fig2 = plt.figure(2)
fig2.set(alpha=0.2)
survived0 = data_train.Pclass[data_train.Survived == 0].value_counts()
survived1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived1, '未获救': survived0})
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'乘客等级')
plt.title(u'各等级存活情况')

fig3 = plt.figure(3)
fig3.set(alpha=0.2)
survived0 = data_train.Sex[data_train.Survived == 0].value_counts()
survived1 = data_train.Sex[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived1, '未获救': survived0})
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'性别')
plt.title(u'男性女性存活情况')

survived0_1 = data_train[data_train.Pclass == 1].Sex[data_train.Survived == 0].value_counts()
survived1_1 = data_train[data_train.Pclass == 1].Sex[data_train.Survived == 1].value_counts()
survived0_2 = data_train[data_train.Pclass == 2].Sex[data_train.Survived == 0].value_counts()
survived1_2 = data_train[data_train.Pclass == 2].Sex[data_train.Survived == 1].value_counts()
survived0_3 = data_train[data_train.Pclass == 3].Sex[data_train.Survived == 0].value_counts()
survived1_3 = data_train[data_train.Pclass == 3].Sex[data_train.Survived == 1].value_counts()

fig4 = plt.figure(4)
fig4.set(alpha=0.2)
df1 = pd.DataFrame({'获救': survived1_1, '未获救': survived0_1})
df1.plot(kind='bar',stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'性别')
plt.title(u'1级乘客生存情况')

fig5 = plt.figure(5)
fig5.set(alpha=0.2)
df2 = pd.DataFrame({'获救': survived1_2, '未获救': survived0_2})
df2.plot(kind='bar',stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'性别')
plt.title(u'2级乘客生存情况')

fig6 = plt.figure(6)
fig6.set(alpha=0.2)
df3 = pd.DataFrame({'获救': survived1_3, '未获救': survived0_3})
df3.plot(kind='bar',stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'性别')
plt.title(u'3级乘客生存情况')

fig7 = plt.figure(7)
fig7.set(alpha=0.2)
survived_have_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
survived_no_cabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({'有': survived_have_cabin, '无': survived_no_cabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'有无船舱')
plt.title(u'有无船舱的生存情况')

fig8 = plt.figure(8)
fig8.set(alpha=0.2)
survived0 = data_train.Embarked[data_train.Survived == 0].value_counts()
survived1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived1, '未获救': survived0})
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'登录港口')
plt.title(u'各个的登录港口的生存情况')

fig9 = plt.figure(9)
fig9.set(alpha=0.2)
survived0 = data_train.SibSp[data_train.Survived == 0].value_counts()
survived1 = data_train.SibSp[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived1, '未获救': survived0})
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'堂兄弟/妹个数')
plt.title(u'堂兄弟/妹个数对生存的影响')

fig10 = plt.figure(10)
fig10.set(alpha=0.2)
survived0 = data_train.Parch[data_train.Survived == 0].value_counts()
survived1 = data_train.Parch[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived1, '未获救': survived0})
df.plot(kind='bar', stacked=True)
plt.ylabel(u'人数')
plt.xlabel(u'父母与小孩个数')
plt.title(u'父母与小孩个数对生存的影响')

plt.show()

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




