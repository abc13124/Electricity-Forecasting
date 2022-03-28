# Electricity-Forecasting

> NCKU DSAI HW1 - Electricity Forecasting

使用台電提供過去的2020、2021年的1至4月的發電量資訊包含淨尖峰供電能力、淨尖峰用電量、民生用電、工業用電等..

並使用Support Vector Regression迴歸模型，預測2022/03/30~04/13的備轉容量(MW)。

## Data Analysis and Feature Selection ##
如[feature.ipynb](https://github.com/abc13124/Electricity-Forecasting/blob/main/feature.ipynb)所示，首先使用pandas.DataFrame.corr以得到各項feature與備轉電力相關的相關係數

```python
corr_mat = df.corr()
mask1 = corr_mat["operating_reserve"] > 0.4
mask2 = corr_mat["operating_reserve"] < -0.4
cap = corr_mat["operating_reserve"]

cols_name = list(cap[mask1 | mask2].index)
print(cols_name)
```

其中，選擇 0.4 作為相關係數，選出較為相關的 12 項係數作為訓練的特徵

> 淨尖峰供電能力(MW)、尖峰負載(MW)、工業用電、民生用電、台中#2、新桃#1、星元#1、嘉惠#1、豐德(#1-#2)、大觀二、明潭與太陽能發電

## Data pre-processing ##
將這些關聯度高的特徵中，刪除2筆在訓練資料集中偏差較大的數值

```python
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

```

## Model training ##
使用scikit-learn中的SVR model，此model設有5種kernel，包含linear, poly, rbf, sigmoid, precomputed。

本次模型參數設定為kernel=poly，Kernel coefficient也就是gamma=0.1、C=1e1。

將2021年1月~4月的資料作為training data,並將資料做Standard Scaler輸入至SVR模型中。

## Run the code ##
環境
Python 3.7.1
```
conda create -n test python==3.7
```
```
activate test
```
路徑移至requirements.txt所在的資料夾，輸入安裝套件指令:
```
conda install --yes --file requirements.txt
```
將app.py、train.csv、test.csv、submission.csv載下後(需在同資料夾內)

輸入以下指令:
```
python app.py --training train.csv --output submission.csv
```
