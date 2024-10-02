import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

my_seed = 3
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from keras.layers import *
from keras.models import *

import time
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['savefig.bbox'] = 'tight'

df = pd.read_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\continuous dataset.csv", index_col=0)
df.index = pd.to_datetime(df.index)

from calendar import day_abbr, month_abbr, mdays

a = day_abbr[0:7]
a.reverse()

data_dict = {0: [df, '', 800, 1400, 2]}
fig, ax = plt.subplots(figsize=(12, 6))
data = data_dict[0][0]
file_name = data_dict[0][1]
hour_week = pd.DataFrame({'nat_demand': data.loc[:, 'nat_demand']})
hour_week['day_of_week'] = hour_week.index.dayofweek
hour_week['hour'] = hour_week.index.hour
hour_week = hour_week.groupby(['day_of_week', 'hour']).mean().unstack()
hour_week = hour_week.iloc[::-1]  # 行反序
hour_week.columns = hour_week.columns.droplevel(0)

sns.heatmap(hour_week, ax=ax, cmap=plt.cm.PuBu, vmax=data_dict[0][3],
            cbar_kws={'boundaries': np.arange(data_dict[0][2], data_dict[0][3], data_dict[0][4])})

cbax = fig.axes[0]
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Day of the week')
ax.set_yticklabels(a)
ax.set_title(data_dict[0][1])

df["time"] = df.index
df['hour of the day'] = df['time'].dt.hour
df['month of the year'] = df['time'].dt.month
df["day of the week"] = df["time"].dt.dayofweek
df['working day'] = df['day of the week'].apply(lambda x: 0 if x > 5 else 1)
df = df.drop(columns="time")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

MO_labels = df.columns[1:]
forest = RandomForestRegressor(max_depth=12, n_estimators=113, random_state=0, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:].values, df["nat_demand"], test_size=0.3,
                                                    shuffle=True, random_state=12)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print(f"{f + 1}) {MO_labels[indices[f]:<30} {importances[indices[f]]}")

import xgboost as xgb

model =
    xgb.XGBRegressor(max_depth=4,
                     learning_rate=0.1,
                     n_estimators=100,
                     objective='reg:linear',
                     booster='gbtree',
                     gamma=0,
                     min_child_weight=0.01,
                     subsample=1,
                     colsample_bytree=0.8,
                     reg_alpha=0,
                     reg_lambda=1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

    xgb.plot_importance(model, max_num_features=11, importance_type='gain', height=0.9)
    im = pd.DataFrame({'importance': model.feature_importances_, 'var': df.columns[1:]})
    im = im.sort_values(by='importance', ascending=False)
    im.head(10)

    im.set_index(im.iloc[:, 1], inplace=False, drop=False)
    im = im.iloc[:, 0]
    t = pd.DataFrame()
    for f in range(X_train.shape[1]):
    t.loc[f, 0] = MO_labels[indices[f]]
    t.loc[f, 1] = importances[indices[f]]

    plt.figure(figsize=[13, 8])
    plt.xticks(rotation=45)
    plt.bar(t.iloc[:,
    0], t.iloc[:, 1])

    t.set_index(t.iloc[:, 0], inplace=False, drop=False)
    t = t.iloc[:, 1].rename("importance")
    feature = t + im
    d = feature.sort_values(ascending=False)

    df = df[['nat_demand', 'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc', 'T2M_san',
             'T2M_dav', 'Holiday_ID', 'holiday', 'hour of the day',
             'month of the year', 'day of the week', 'working day']]
    df.to_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\筛选后的负荷与其它特征.csv", sep=',', index=True, header=True,
              encoding='utf-8-sig')
    d.to_csv(r"E:\桌面\活\活2（python）\特征选择-csv文件\特征重要性.csv", sep=',', index=True, header=True,
             encoding='utf-8-sig')
