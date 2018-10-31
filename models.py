# -*- coding: utf-8 -*-

'''
Author: Bijan Fallah 
Code for AUTO1 interview 
2018
'''
#=================================================  importing 
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
import  matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
# ================================================ name list
rand = int(sys.argv[1])
os.system('python preprocessing.py')
#================================================= load data

X_train = pd.read_pickle('./X_train_nopca')
X_test = pd.read_pickle('./X_test_nopca')
y_train = pd.read_pickle('./y_train_nopca')
y_test = pd.read_pickle('./y_test_nopca')


#=============================================== split for layer 1

X_train_stack, X_train_hold_out, y_train_stack, y_train_hold_out = train_test_split(X_train,
                                                                                    y_train,
                                                                                    test_size=0.2,
                                                                                    random_state=rand)

# ============== train models on train data

# based on experiments done before I choose trhe best hyper parameters!



## Ridge
print('--------Ridge---------')
#param_grid = [{'alpha': np.logspace(-2, 2, 50)}]
#ridge_reg = linear_model.Ridge(random_state=rand)
#reg_ridge = GridSearchCV(ridge_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
reg_ridge = linear_model.Ridge(random_state=rand, alpha =  0.9102981779915218 )
reg_ridge.fit (X_train_stack, y_train_stack)
#print(reg_ridge.best_params_)


## LASSO
print('--------Lasso ---------')

#param_grid = [{'alpha': np.logspace(-2, 2, 50)}]
#lasso_reg = linear_model.Lasso(random_state=rand, tol=1e-2)
#reg_lasso= GridSearchCV(lasso_reg, param_grid, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
reg_lasso= linear_model.Lasso(random_state=rand, tol=1e-2, alpha= 0.16768329368110074)
reg_lasso.fit (X_train_stack, y_train_stack)
#print(reg_lasso.best_params_)
print('--------Elastic net ---------')


## ElasticNet
#param_grid = [{'alpha': np.logspace(-2, 2, 50), 'l1_ratio': np.linspace(0.1, 0.98, 40)}]
#elastic_net_reg = ElasticNet(tol=1e-2,random_state=rand)
#reg_elastic = GridSearchCV(elastic_net_reg, param_grid, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
reg_elastic = ElasticNet(tol=1e-2,random_state=rand, alpha = 0.13894, l1_ratio = 0.98)
reg_elastic.fit(X_train_stack,y_train_stack)
#print(reg_elastic.best_params_)


## Random Forest
print('-------- RF ---------')
#param_grid = [{'n_estimators': np.arange(10,50), 'max_depth': np.arange(3, 20),
#               'max_features': np.arange(15, 16)}]
#rf_reg = RandomForestRegressor(criterion='mse',random_state=rand)
#reg_rf = GridSearchCV(rf_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
reg_rf = RandomForestRegressor(criterion='mse',random_state=rand, n_estimators = 31, max_depth=14, max_features = 15)

reg_rf.fit(X_train_stack,y_train_stack)
#print(reg_rf.best_params_)


## XGBoost
print('--------XGBoost ---------')
#param_grid = [{'gamma': [0.1, 0.5, 1, 1.5, 2, 5], 'max_depth': np.arange(3, 7)}]
#xgb_reg = XGBRegressor(random_state=rand)
#reg_xgb = GridSearchCV(xgb_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
reg_xgb =  XGBRegressor(random_state=rand, max_depth = 3, gamma= 0.5)
reg_xgb.fit(X_train_stack,y_train_stack)
#print(reg_xgb.best_params_)
# ===================== predict layer 1 input  for stacking

train_blend_in_ridge = reg_ridge.predict(X_train_hold_out)
train_blend_in_lasso = reg_lasso.predict(X_train_hold_out)
train_blend_in_elast = reg_elastic.predict(X_train_hold_out)
train_blend_in_randf = reg_rf.predict(X_train_hold_out)
train_blend_in_xgbst = reg_xgb.predict(X_train_hold_out)
print("=========================================================")



# ===================== predict layer 1 test input for stacking 

test_blend_in_ridge = reg_ridge.predict(X_test)
test_blend_in_lasso = reg_lasso.predict(X_test)
test_blend_in_elast = reg_elastic.predict(X_test)
test_blend_in_randf = reg_rf.predict(X_test)
test_blend_in_xgbst = reg_xgb.predict(X_test)


print("ridge Test Error (RMSE): ", np.sqrt(mean_squared_error(y_test,test_blend_in_ridge)))
print("lasso Test Error (RMSE): ", np.sqrt(mean_squared_error(y_test,test_blend_in_lasso)))
print("elastic Test Error (RMSE): ", np.sqrt(mean_squared_error(y_test,test_blend_in_elast)))
print("random forest Test Error (RMSE): ", np.sqrt(mean_squared_error(y_test,test_blend_in_randf)))
print("xgboost Test Error (RMSE): ", np.sqrt(mean_squared_error(y_test,test_blend_in_xgbst)))


# blend them all
train_input = pd.DataFrame({'ridge': train_blend_in_ridge, 'lasso': train_blend_in_lasso,
                            'elast':train_blend_in_elast, 'randf': train_blend_in_randf,
                            'xgnst':train_blend_in_xgbst})
train_out = y_train_hold_out
test_input = pd.DataFrame({'ridge': test_blend_in_ridge, 'lasso': test_blend_in_lasso,
                            'elast':test_blend_in_elast, 'randf': test_blend_in_randf,
                            'xgnst':test_blend_in_xgbst})
test_out = y_test


# I will check 3 different stacking methods :



def ensemble_average(X_train, y_train, X_test, y_test, report=True):
    """Returns y_true, y_test"""
    y_train_pred = X_train.apply(lambda x: x.mean(), axis=1)
    y_pred = X_test.apply(lambda x: x.mean(), axis=1)

    if report:
        print("\n\n========== [Stacking Report: Averaging] ==========")
        mse = mean_squared_error(y_test, y_pred)
        print("AVG Stacking Test Error(RMSE)    : ", np.sqrt(mse))
    return y_test, y_pred


def linear_regression_stacking(X_train, y_train, X_test, y_test, report=True):
    """Returns linear regression blender model"""
    blender = LinearRegression()
    blender.fit(X_train, y_train)
    y_train_pred = blender.predict(X_train)
    y_pred = blender.predict(X_test)
    if report:
        print("\n\n========== [Stacking Report: Linear Regression stacking] ==========")
        mse = mean_squared_error(y_test, y_pred)
        print("Linear reg. Stacking Test Error(RMSE)    : ", np.sqrt(mse))

        

    return y_test, y_pred

def NN_stacking(X_train, y_train, X_test, y_test, report=True):
    """Returns neural networks  model"""

    from tensorflow.layers import Dense, BatchNormalization
    from tensorflow.keras import Sequential
    from tensorflow import losses
    from tensorflow import set_random_seed
    set_random_seed(rand)
    blender = Sequential()
    blender.add(Dense(3, input_shape=(5, )))
    blender.add(BatchNormalization())
    blender.add(Dense(1, activation='relu'))
    blender.compile(optimizer='adam', loss=losses.mean_squared_error)
    blender.fit(np.array(X_train), np.array(y_train), epochs=3000, verbose=0)
    y_train_pred = blender.predict(np.array(X_train))
    y_pred = blender.predict(np.array(X_test))

    if report:
        print("\n\n========== [Stacking Report: NN stacking] ==========")
        mse = mean_squared_error(y_test, y_pred)
        print("NN Stacking Test Error(RMSE)    : ", np.sqrt(mse))

        

    return y_test, y_pred






y_test_av, y_train_av = ensemble_average(train_input, train_out, test_input, test_out)
y_test_lm, y_train_lm =  linear_regression_stacking(train_input, train_out, test_input, test_out)
y_test_nn, y_train_nn =  NN_stacking(train_input, train_out, test_input, test_out)

#fig, ax = plt.subplots()
#ax.plot(y_test_av, y_train_av,'go',alpha=.6, label='AV')
#ax.plot(y_test_lm, y_train_lm,'ko', alpha=.6, label='LM')
#ax.plot(y_test_nn, y_train_nn,'bo', alpha=.6, label='NN')
#ax.legend()
#ax.plot(np.arange(300), np.arange(300),'k--')
#ax.set_xlim((50,255))
#ax.set_ylim((50,255))
#ax.axis('equal')
#plt.show()
