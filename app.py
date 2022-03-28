# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:41:23 2022

@author: Guan Lin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import metrics
import seaborn as sns  #繪圖

def feature(train):
    df = pd.read_csv("train.csv")
    # df.head(10).append(df.tail(10))

    corr_mat = df.corr()
    mask1 = corr_mat["operating_reserve"] > 0.4
    mask2 = corr_mat["operating_reserve"] < -0.4
    cap = corr_mat["operating_reserve"]
    cols_name = list(cap[mask1 | mask2].index)
    
    #將顯示前一筆最大的數據，並將其刪除，試圖減少偏差值
    train=train.drop(index=train.sort_values(by='ele_pro',ascending=False)[:2].index)
    train=train.drop(index=train.sort_values(by='operating_reserve',ascending=False)[:2].index)
    train=train.drop(index=train.sort_values(by='ele',ascending=False)[:2].index)
    train=train.drop(index=train.sort_values(by='zintao1',ascending=False)[:2].index)
    train=train.drop(index=train.sort_values(by='emp',ascending=False)[:2].index)
    train=train.drop(index=train.sort_values(by='life',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='TC',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='star',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='chia',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='FD',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='DG',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='MT',ascending=False)[:2].index)  
    train=train.drop(index=train.sort_values(by='sun',ascending=False)[:2].index)  

    return train


def sub(pred):
    name = ['operating_reserve']
    pred = pd.DataFrame(pred, columns=name)
    date = [['date'],['2022/3/30'],['2022/3/31'],['2022/4/01'],
            ['2022/4/01'],['2022/4/02'],['2022/4/03'],['2022/4/04'],
            ['2022/4/05'],['2022/4/06'],['2022/4/07'],['2022/4/08'],
            ['2022/4/09'],['2022/4/10'],['2022/4/11'],['2022/4/12'],['2022/4/13']]
    name = date.pop(0)
    date_df = pd.DataFrame(date,columns=name)
    res = pd.concat([date_df,pred],axis = 1)
    res.to_csv(args.output,index=0, float_format='%.3f')

def forecasting():
    #import data
    train = pd.read_csv(args.training)
    train_new = feature(train)

    #建立training dataset
    X = train_new[['ele_pro', 'ele','emp','zintao1','life','TC','star','chia','FD','DG','MT','sun']]

    Y = train_new[['operating_reserve']]
    Y = Y.values.reshape(-1,1) 
    
    #Feature scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_x = scaler_x.fit_transform(X)
    train_y = scaler_y.fit_transform(Y)
    #SVR
    regressor = SVR(kernel='poly', C=1e1, gamma=0.01)
    regressor.fit(train_x,train_y)
    
    test = pd.read_csv('test.csv')
    
    pred_x = test[['ele_pro', 'ele','emp','zintao1','life','TC','star','chia','FD','DG','MT','sun']][22:38]
    pred_x = scaler_x.fit_transform(pred_x)
    pred = regressor.predict(pred_x)
    pred =scaler_y.inverse_transform(pred)
    sub(pred)
    # write the result
 

if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='train.csv',
                        help='input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    forecasting()