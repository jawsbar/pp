import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import skew

def scatter(data, var):
    matplotlib.rcParams['figure.figsize'] = (12.0, 4.0)
    data = pd.concat([data['SalePrice'], data[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


data = pd.read_csv("train.csv")
matplotlib.rcParams['figure.figsize'] = (16.0, 6.0)
prices = pd.DataFrame({"log(price+1)":np.log1p(data["SalePrice"]), "price":data["SalePrice"]})
prices.hist()

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
corrmat = data.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
scatter(data, 'GrLivArea')



def load_file(is_test=False):
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")

    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

    data_df = pd.concat((train.loc[:, cols], test.loc[:, cols]))

    numeric_feats = data_df.dtypes[data_df.dtypes != 'object'].index
    skewed_feats = data_df[numeric_feats].apply(lambda x:skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    data_df[skewed_feats] = np.log1p(data_df[skewed_feats])
    data_df = pd.get_dummies(data_df)
    data_df = data_df.fillna(data_df.mean())

    return data_df, train, test

all_data, train, test = load_file()
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y_train = np.expand_dims(np.log1p(train.SalePrice), axis=1)

print(x_train.shape, y_train.shape)

print(all_data.head())