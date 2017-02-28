import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation


# Import libraries
import numpy as np
import collections
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
rcParams['figure.figsize']= 15,6
import timeit

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from pylab import *

from sklearn import cross_validation
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.isotonic import IsotonicRegression

from sklearn.preprocessing import PolynomialFeatures


def plotFeatures(data):
    rcParams['figure.figsize'] = (15, 20)
    
    plt.subplot(511)
    data['SST'].plot();
    # Add text
    plt.title('Sea Surface Temperature')
    plt.xlabel('Observations')
    plt.ylabel('Sea Surface Temperature [degree]')
    plt.tight_layout()
    
    plt.subplot(512)
    data['AT'].plot();
    # Add text
    plt.title('Air Temperature')
    plt.xlabel('Observations')
    plt.ylabel('Air Temperature [degree]')
    plt.tight_layout()
    
    plt.subplot(513)
    data['Humid'].plot();
    # Add text
    plt.title('Humidity')
    plt.xlabel('Observations')
    plt.ylabel('Humidity [%]')
    plt.tight_layout()
    
    plt.subplot(514)
    data['MW'].plot();
    # Add text
    plt.title('Meridional Winds')
    plt.xlabel('Observations')
    plt.ylabel('Meridional Winds [m/s]')
    plt.tight_layout()
    
    plt.subplot(515)
    data['ZW'].plot();
    # Add text
    plt.title('Zonal Winds')
    plt.xlabel('Observations')
    plt.ylabel('Zonal Winds [m/s]')
    plt.tight_layout()
    
    
def plotGeneralTendencies(data):
    rcParams['figure.figsize'] = (15, 20)
    year_uniq = data['Yr'].unique();

    plt.subplot(511)
    std_SST = data.groupby(['Yr'], as_index=False).agg({'SST' : 'std'})['SST']
    mean_SST = data.groupby(['Yr'], as_index=False).agg({'SST' : 'mean'})['SST']

    plt.errorbar(year_uniq, mean_SST, yerr=std_SST, capsize=8, elinewidth=1)

    # Add text
    plt.title('Evolution of Sea Surface Temperature')
    plt.xlabel('Years')
    plt.ylabel('Sea Surface Temperature [degree]')
    plt.tight_layout()

    plt.subplot(512)
    std_AT = data.groupby(['Yr'], as_index=False).agg({'AT' : 'std'})['AT']
    mean_AT = data.groupby(['Yr'], as_index=False).agg({'AT' : 'mean'})['AT']

    plt.errorbar(year_uniq, mean_AT, yerr=std_AT, capsize=8, elinewidth=1)

    # Add text
    plt.title('Evolution of Air Temperature')
    plt.xlabel('Years')
    plt.ylabel('Air Temperature [degree]')
    plt.tight_layout()

    plt.subplot(513)
    std_Humid = data.groupby(['Yr'], as_index=False).agg({'Humid' : 'std'})['Humid']
    mean_Humid = data.groupby(['Yr'], as_index=False).agg({'Humid' : 'mean'})['Humid']

    plt.errorbar(year_uniq, mean_Humid, yerr=std_Humid, capsize=8, elinewidth=1)

    # Add text
    plt.title('Evolution of Humidity')
    plt.xlabel('Years')
    plt.ylabel('Humidity [%]')
    plt.tight_layout()

    plt.subplot(514)
    mean_MW = data.groupby(['Yr'], as_index=False).agg({'MW' : 'std'})['MW']
    std_MW = data.groupby(['Yr'], as_index=False).agg({'MW' : 'mean'})['MW']

    plt.errorbar(year_uniq, mean_MW, yerr=std_MW, capsize=8, elinewidth=1)

    # Add text
    plt.title('Evolution of Meridional Winds')
    plt.xlabel('Years')
    plt.ylabel('Meridional Winds [m/s]')
    plt.tight_layout()

    plt.subplot(515)
    std_ZW = data.groupby(['Yr'], as_index=False).agg({'ZW' : 'std'})['ZW']
    mean_ZW = data.groupby(['Yr'], as_index=False).agg({'ZW' : 'mean'})['ZW']

    plt.errorbar(year_uniq, mean_ZW, yerr=std_ZW, capsize=8, elinewidth=1)

    # Add text
    plt.title('Evolution of Zonal Winds')
    plt.xlabel('Years')
    plt.ylabel('Zonal Winds [m/s]')
    plt.tight_layout()
    
    
    
def plotFeaturesCorrelation(data):
    rcParams['figure.figsize'] = (15, 20)
    plt.subplot(511)
    #plt.subplot(figsize=(15,15))
    group = data.groupby('SST').mean()
    corr = data['AT'].corr(data['SST'], method='pearson')
    trace1 = group['AT'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,17));
    plt.xlabel('Sea Surface Temperature [degree]')
    plt.ylabel('Air Temperature [degree]')
    #fig.set_size_inches(15, 10)
    plt.tight_layout()

    plt.subplot(512)
    group = data.groupby('ZW').mean()
    corr = data['MW'].corr(data['ZW'], method='pearson')
    trace2 = group['MW'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,17));
    plt.xlabel('Zonal winds [m/s]')
    plt.ylabel('Meridional winds [m/s]')
    plt.tight_layout()

    plt.subplot(513)
    group = data.groupby('AT').mean()
    corr = data['MW'].corr(data['AT'], method='pearson')
    trace3 = group['MW'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,17));
    plt.xlabel('Air Temperature [degree]')
    plt.ylabel('Meridional winds [m/s]')
    plt.tight_layout()

    plt.subplot(514)
    group = data.groupby('AT').mean()
    corr = data['ZW'].corr(data['AT'], method='pearson')
    trace4 = group['ZW'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,17));
    plt.xlabel('Air Temperature [degree]')
    plt.ylabel('Zonal winds [m/s]')
    plt.tight_layout()

    plt.subplot(515)
    group = data.groupby('AT').mean()
    corr = data['Humid'].corr(data['AT'], method='pearson')
    group['Humid'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,17));
    plt.xlabel('Air Temperature [degree]')
    plt.ylabel('Humidity [%]')
    
    
# Plot the correlation between humidity and winds
def plotCorrelationHW(data):
    # Correlation between Humidity and Meridional winds:
    plt.subplot(211)
    group = data.groupby('Humid').mean()
    corr = data['MW'].corr(data['Humid'], method='pearson')
    group['MW'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,10));
    plt.xlabel('Humidity [%]')
    plt.ylabel('Meridional winds [m/s]')
    plt.tight_layout()

    plt.subplot(212)
    group = data.groupby('Humid').mean()
    corr = data['ZW'].corr(data['Humid'], method='pearson')
    group['ZW'].plot(grid=True, title='Pearson correlation: {:.4f}'.format(corr), figsize=(15,10));
    plt.xlabel('Humidity [%]')
    plt.ylabel('Zonal winds [m/s]')
    plt.tight_layout()
    
    

# Checking stationarity of TS with rolling mean and std, and Dickey-Fuller test
# If Test statistic < Critical value --> TS is stationary
def test_stationarity(timeseries):
    rcParams['figure.figsize'] = (15, 7)

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    #rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = timeseries.rolling(window=12).std()
    #rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
#ACF and PACF plots:    
def plotACF_PACF(timeseries): 
    lag_acf = acf(timeseries, nlags=20)
    lag_pacf = pacf(timeseries, nlags=20, method='ols')

    #Plot Auto-Correlation Function (ACF): 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot Partial Autocorrelation Function (PACF):
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
def plotResARMA(test,predictions):
    df1 = pd.DataFrame({'Predictions': predictions})
    df1.index = test.index
    predictions = df1['Predictions']

    plt.plot(test,'-o',label='Test set',markersize = 7)
    plt.plot(predictions,'--^', color='red',label='Predictions',markersize = 7)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
    
# Modify the function to plot the years dynamicaly
def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)
        

def baseMapResult(data):
    longitude = np.array(data['Long'])
    latitude = np.array(data['Lat'])
    longitude_round = np.around(longitude, decimals=-1)
    latitude_round = np.around(np.array(data['Lat']))
    airTemperature = np.array(data['AT'])
    years = np.array(data['Yr'])
    longitude_unique = np.unique(longitude)
    latitude_unique = np.unique(latitude)
    longitude_rnd_unique = np.unique(longitude_round)
    latitude_rnd_unique = np.unique(latitude_round)
    
    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.
    map = Basemap(projection='hammer',lat_0=0,lon_0=-160,resolution='l')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # compute native map projection coordinates of lat/lon grid.
    # contour data over the map.
    map.scatter(longitude,latitude,latlon = True,marker='o',color='r')
    #plt.title("Flickr Geotagging Counts with Basemap")
    plt.title('Buoys positions from 1980 to 1998')
    plt.show()
    
    map = Basemap(projection='hammer',lat_0=0,lon_0=-160,resolution='l')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines
    map.drawmeridians(longitude_rnd_unique)
    map.drawparallels(latitude_rnd_unique)
    map.scatter(longitude_round,latitude_round,latlon = True,marker='o',color='r')
    plt.title('Buoys per zone')
    plt.show()
    return longitude, latitude, years


def RegressLin(X_train,X_test,Y_train):
    # Linear regression
    lm = LinearRegression()

    start_timeFit = timeit.default_timer()
    lm.fit(X_train,Y_train)
    stop_timeFit = timeit.default_timer()
    timeLin = stop_timeFit - start_timeFit

    trainPredLin = lm.predict(X_train)
    testPredLin = lm.predict(X_test)
    return timeLin, trainPredLin, testPredLin

def RegressLasso(X_train,X_test,Y_train, alpha):
    # Lasso regression
    #alpha = 1e-3

    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)

    start_timeLas = timeit.default_timer()
    lassoreg.fit(X_train,Y_train)
    stop_timeLas = timeit.default_timer()
    timeLas = stop_timeLas-start_timeLas

    trainPredLas = lassoreg.predict(X_train)
    testPredLas = lassoreg.predict(X_test)
    return timeLas, trainPredLas, testPredLas

def RegressRidge(X_train,X_test,Y_train):
    #Ridge regression
    ridge = linear_model.Ridge()

    start_timeRidge = timeit.default_timer()
    ridge.fit(X_train, Y_train)
    stop_timeRidge = timeit.default_timer()
    timeRidge = stop_timeRidge-start_timeRidge

    trainPredRidge = ridge.predict(X_train)
    testPredRidge = ridge.predict(X_test)
    return timeRidge, trainPredRidge, testPredRidge

def RegressElasticNet(X_train,X_test,Y_train):
    #ElasticNet regression
    elasticNet = linear_model.ElasticNet()

    start_timeElasticNet = timeit.default_timer()
    elasticNet.fit(X_train, Y_train)
    stop_timeElasticNet = timeit.default_timer()
    timeElastic = stop_timeElasticNet-start_timeElasticNet

    trainPredElastic = elasticNet.predict(X_train)
    testPredElastic = elasticNet.predict(X_test)
    return timeElastic, trainPredElastic, testPredElastic

def RegressPoly(X_train,X_test,Y_train, order):
    # Ploynomial of degree #order
    clf = linear_model.LinearRegression()

    start_timePoly = timeit.default_timer()
    poly = PolynomialFeatures(degree=order)
    X_train_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)
    clf.fit(X_train_, Y_train)
    stop_timePoly= timeit.default_timer()
    timePoly = stop_timePoly-start_timePoly

    trainPredPoly = clf.predict(X_train_)
    testPredPoly = clf.predict(X_test_)
    return timePoly, trainPredPoly, testPredPoly


def metricDisplay(Y_train,Y_test,trainPredLin,testPredLin,trainPredLas,testPredLas,trainPredRidge,testPredRidge,trainPredElastic,testPredElastic,trainPredPoly5,testPredPoly5,trainPredPoly4,testPredPoly4,trainPredPoly3,testPredPoly3,trainPredPoly2,testPredPoly2):
    # Metric computation
    #Linear
    MSETrainLin = mean_squared_error(trainPredLin, np.array(Y_train).astype(np.float))
    MSETestLin = mean_squared_error(testPredLin, np.array(Y_test).astype(np.float))
    R2TrainLin = r2_score(np.array(Y_train).astype(np.float),trainPredLin)
    R2TestLin = r2_score( np.array(Y_test).astype(np.float),testPredLin)

    #LASSO
    MSETrainLas = mean_squared_error(trainPredLas, np.array(Y_train).astype(np.float))
    MSETestLas = mean_squared_error(testPredLas, np.array(Y_test).astype(np.float))
    R2TrainLas = r2_score(np.array(Y_train).astype(np.float),trainPredLas)
    R2TestLas = r2_score(np.array(Y_test).astype(np.float),testPredLas)

    #Ridge
    MSETrainRidge = mean_squared_error(trainPredRidge, np.array(Y_train).astype(np.float))
    MSETestRidge = mean_squared_error(testPredRidge, np.array(Y_test).astype(np.float))
    R2TrainRidge = r2_score(np.array(Y_train).astype(np.float),trainPredRidge)
    R2TestRidge = r2_score(np.array(Y_test).astype(np.float),testPredRidge)

    #ElasticNet
    MSETrainElastic = mean_squared_error(trainPredElastic, np.array(Y_train).astype(np.float))
    MSETestElastic = mean_squared_error(testPredElastic, np.array(Y_test).astype(np.float))
    R2TrainElastic = r2_score(np.array(Y_train).astype(np.float),trainPredElastic)
    R2TestElastic = r2_score(np.array(Y_test).astype(np.float),testPredElastic)

    #Ploynomial degree 5 
    MSETrainPoly5 = mean_squared_error(trainPredPoly5, np.array(Y_train).astype(np.float))
    MSETestPoly5 = mean_squared_error(testPredPoly5, np.array(Y_test).astype(np.float))
    R2TrainPoly5 = r2_score(np.array(Y_train).astype(np.float),trainPredPoly5)
    R2TestPoly5 = r2_score(np.array(Y_test).astype(np.float),testPredPoly5)

    #Ploynomial degree 4
    MSETrainPoly4 = mean_squared_error(trainPredPoly4, np.array(Y_train).astype(np.float))
    MSETestPoly4 = mean_squared_error(testPredPoly4, np.array(Y_test).astype(np.float))
    R2TrainPoly4 = r2_score(np.array(Y_train).astype(np.float),trainPredPoly4)
    R2TestPoly4 = r2_score(np.array(Y_test).astype(np.float),testPredPoly4)

    #Ploynomial degree 3
    MSETrainPoly3 = mean_squared_error(trainPredPoly3, np.array(Y_train).astype(np.float))
    MSETestPoly3 = mean_squared_error(testPredPoly3, np.array(Y_test).astype(np.float))
    R2TrainPoly3 = r2_score(np.array(Y_train).astype(np.float),trainPredPoly3)
    R2TestPoly3 = r2_score(np.array(Y_test).astype(np.float),testPredPoly3)

    #Ploynomial degree 2
    MSETrainPoly2 = mean_squared_error(trainPredPoly2, np.array(Y_train).astype(np.float))
    MSETestPoly2 = mean_squared_error(testPredPoly2, np.array(Y_test).astype(np.float))
    R2TrainPoly2 = r2_score(np.array(Y_train).astype(np.float),trainPredPoly2)
    R2TestPoly2 = r2_score(np.array(Y_test).astype(np.float),testPredPoly2)

    # Display Metric
    print('Results with respect to several regressions : ')
    print('----------------------------------------------------------------------------------------')
    print('Linear:')
    print('Mean squared error, train set : ',MSETrainLin,', test set : ', MSETestLin)
    print('R2 score, train set : ',R2TrainLin,', test set : ',R2TrainLin)
    print('----------------------------------------------------------------------------------------')
    print('Lasso:')
    print('Mean squared error, train set : ',MSETrainLas,', test set : ', MSETestLas)
    print('R2 score, train set : ',R2TrainLas,', test set : ',R2TrainLas)
    print('----------------------------------------------------------------------------------------')
    print('Ridge:')
    print('Mean squared error, train set : ',MSETrainRidge,', test set : ', MSETestRidge)
    print('R2 score, train set : ',R2TrainRidge,', test set : ',R2TrainRidge)
    print('----------------------------------------------------------------------------------------')
    print('ElasticNet:')
    print('Mean squared error, train set : ',MSETrainElastic,', test set : ', MSETestElastic)
    print('R2 score, train set : ',R2TrainElastic,', test set : ',R2TrainElastic)
    print('----------------------------------------------------------------------------------------')
    print('Polynomial degree 2:')
    print('Mean squared error, train set : ',MSETrainPoly2,', test set : ', MSETestPoly2)
    print('R2 score, train set : ',R2TrainPoly2,', test set : ',R2TrainPoly2)
    print('----------------------------------------------------------------------------------------')
    print('Polynomial degree 3:')
    print('Mean squared error, train set : ',MSETrainPoly3,', test set : ', MSETestPoly3)
    print('R2 score, train set : ',R2TrainPoly3,', test set : ',R2TrainPoly3)
    print('----------------------------------------------------------------------------------------')
    print('Polynomial degree 4:')
    print('Mean squared error, train set : ',MSETrainPoly4,', test set : ', MSETestPoly4)
    print('R2 score, train set : ',R2TrainPoly4,', test set : ',R2TrainPoly4)
    print('----------------------------------------------------------------------------------------')
    print('Polynomial degree 5:')
    print('Mean squared error, train set : ',MSETrainPoly5,', test set : ', MSETestPoly5)
    print('R2 score, train set : ',R2TrainPoly5,', test set : ',R2TrainPoly5)