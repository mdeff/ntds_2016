import pandas as pd
import os.path
import matplotlib.pyplot as plt

def makeTimeSeries(df):
    ts = pd.to_datetime(df.dt)
    df.index = ts
    return df.drop('dt', axis=1)

def differenciate(X):
    diff = list()
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)
    X_diff=pd.DataFrame(diff)
    X_diff.index=X.index[1:]
    X_diff=X_diff[0]
    return X_diff

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=10)
    rolstd = pd.rolling_std(timeseries, window=10)

    #Plot rolling statistics:
    plt.figure(figsize=(16,8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.xlabel('years',fontsize=16)
    plt.ylabel('Temperature, °C',fontsize=16)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation',fontsize=24)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)