# -*- coding: utf-8 -*-
'''
Author : Bijan 
code for auto1 interview
2018

'''
# =================== importing of libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ==================== name-list 
rand = 777 # for reproducability 
pca_flag=False

#===================== Load the data % data engineering, cleaning 


data = pd.read_csv("Auto1-DS-TestData.csv") 

# nans
data = data.replace('?', np.NaN)
data = data.replace('',np.NaN)
data = data.replace('NA',np.NaN)
# replace - with _ in the names
names = []
for i in data.columns:
    i = i.replace('-','_')
    names.append(i)
# convert some strings to numbers :
data.columns = names
data_no_na = data.dropna() # remove rows with nans (no imputing)
# drop the actuaries estimate for the symboling
data_cleaned = data_no_na.drop(['symboling'], axis=1)


X = data_cleaned.copy()
mapping_num_of_doors = pd.Series([2, 4], index=['two', 'four'])
mapping_num_of_cylinders = pd.Series([2, 3, 4, 5, 6, 8, 12], index=['two', 'three', 'four', 'five', 'six', 'eight', 'twelve'])
X.loc[:, 'num_of_doors'] = X.loc[:, 'num_of_doors'].map(mapping_num_of_doors)
X.loc[:, 'num_of_cylinders'] = X.loc[:, 'num_of_cylinders'].map(mapping_num_of_cylinders)

# treat categorical variables 

features_no = ['num_of_doors', 'wheel_base', 'length', 'width', 'height',
                'curb_weight', 'num_of_cylinders', 'engine_size', 'bore', 'stroke', 'compression_ratio',
                'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
features_cat = ['make', 'fuel_type', 'aspiration', 'body_style', 'drive_wheels','engine_type', 'fuel_system']
# str to numerics :
for col in X.columns:
    if col in features_no:
        X.loc[:, col] = pd.to_numeric(X.loc[:, col])
    if col in features_cat:
        X.loc[:, col] = X.loc[:, col].astype('category')
        


        
# drop engine location : 
#X = X.drop(['engine_location'], axis=1)
X_onehot = X.copy()
X_OHE = pd.get_dummies(X_onehot[features_cat])
X_nums = X_onehot[features_no]
scaler = StandardScaler()
X_f = pd.DataFrame(scaler.fit_transform(X_nums), index=X_nums.index, columns=X_nums.columns)
X_final = pd.concat([X_OHE,X_f,X['normalized_losses'] ], axis=1)
print(X_final.head())



def find_correlation(data, threshold=0.85, remove_negative=False):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.
    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value.
    remove_negative: Boolean
        If true then features which are highly negatively correlated will
        also be returned for removal.
    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat
    
# find list of features with high correlation
    
lists = find_correlation(X_final, threshold=0.70, remove_negative=False)
    

# do pca on highly correlated features 
lists.append('length')
lists.append('fuel_type_gas')
lists.append('compression_ratio')
lists.append('engine_size')
lists.append('highway_mpg')
lists.append('horsepower')
lists.append('wheel_base')
my_model = PCA(n_components=10)
pcas = my_model.fit_transform(X_final[lists])

XX_pca = X_final.drop(lists, axis = 1)
XX_pca['pc1'] = pcas[:,0]
XX_pca['pc2'] = pcas[:,1]

# split to x and y :
if pca_flag:
    print("PCA--------")
    y = pd.to_numeric(XX_pca["normalized_losses"])
    X_f = XX_pca.drop("normalized_losses", axis=1)
else:
    
    y = pd.to_numeric(X_final["normalized_losses"])
    X_f = X_final.drop("normalized_losses", axis=1)
    
# split data to train and test 



X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size=0.2,
                                                    random_state=rand)
                                                    
                                                    
# save the clean data: 

X_train.to_pickle('X_train_nopca')
y_train.to_pickle('y_train_nopca')

X_test.to_pickle('X_test_nopca')
y_test.to_pickle('y_test_nopca')


print('preprocessing finished!')
print(X_train.shape)



                                            



