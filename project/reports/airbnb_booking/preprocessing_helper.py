# Functions to preprocess Train_user and Test_user
# Pecoraro Cyril

import pandas as pd
import numpy as np
import os
from IPython.display import Image
from IPython.core.display import HTML 
import matplotlib.pyplot as plt  
import random
from datetime import datetime
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import copy
import pylab
import calendar



# clean age
# @arg(in) df : DataFrame
# @arg(in) type : 'k' : unvalid data to be replaced by -1. 'd' : unvalid data to be deleted
def cleanAge(df, type):
    # Finding users who put their birthdate instead of age in original dataframe
    df_birthyear = df[(df['age']>=1916) & (df['age']<=2001)]

    # Converting to age
    df_birthyear = copy.deepcopy(df_birthyear)
    df_birthyear['age'] = 2016-df_birthyear['age']

    # Replacing in original dataframe
    df.loc[(df['age']>=1926) & (df['age']<=2001), 'age'] = df_birthyear


    # Assigning a -1 value to invalid ages
    df = copy.deepcopy(df)
    df.loc[((df['age']<15) | (df['age']>100)), 'age'] = -1
    
    if(type == 'k'):
        # Counting invalid ages:
        OutOfBoundsAgePercentage = round(100*len(df.loc[(df['age'] == -1), 'age'])/len(df),2)
        print('Percentage of users with irrelevant age',OutOfBoundsAgePercentage,'%')

        # Counting NaN ages:
        nanAgePercentage = round(100*df['age'].isnull().values.sum()/len(df),2)
        print('Percentage of users with NaN age',nanAgePercentage,'%')

        # Assigning a -1 value to NaN ages
        df = copy.deepcopy(df)
        df['age'].fillna(-1, inplace = True) 
        print('All the invalid or missing age were replaced by value -1')
        
    if(type == 'd'):
        df = df[df['age'] != -1]
    df['age'] = df['age'].astype(int)
    return df

# clean ageBucket
# @arg(in) df : DataFrame
def cleanAgeBucket(df):
    df.drop(df.ix[df['age_bucket'].isin(['0-4','5-9','10-14','100+'])].index, inplace= True)
    return df

# preprocess display of travelers per country given age and sex
# @arg(in) df : DataFrame
def travellerCountryProcess(df):
    # remove  year
    if 'year' in df:
        df.drop(['year'],axis=1,inplace = True)

    # Compute number of people by age (previously, people were characterized by age AND sex)
    df_destination_age = df.groupby(['country_destination','age_bucket','gender'],as_index=False).sum()

    # Compute total number of people by country 
    df_destination_total = df_destination_age.groupby('country_destination').sum().reset_index()

    # Incorpore total in the df_destination_age dataframe
    df_destination_age = df_destination_age.merge(df_destination_total, on='country_destination')
    df_destination_age=df_destination_age.rename(columns = {'population_in_thousands_y':'population_total_in_thousands',
                                                            'population_in_thousands_x':'population_in_thousands'})

    # Compute share of people by age and destination 
    df_destination_age['proportion_%']=np.round(df_destination_age['population_in_thousands']/
                                                 df_destination_age['population_total_in_thousands']*100, 
                                                 decimals=1)

    # Index dataframe by country of destination then age
    df_destination_age_male = df_destination_age.loc[df_destination_age['gender'] == 'male']
    df_destination_age_female = df_destination_age.loc[df_destination_age['gender'] == 'female']
    
    return df_destination_age_male,df_destination_age_female,df_destination_total
    
# Display of travelers proportion per country given age and sex
# @arg(in) df_destination_age_male : DataFrame  of males  
# @arg(in) df_destination_age_female : DataFrame  of females  
def travellerProportionCountryPlot(df_destination_age_male,df_destination_age_female):
    SIZE = 10
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    #male in blue
    fig, axes = plt.subplots(nrows=5, ncols=2)
    fig.set_size_inches(10, 12, forward=True)

    for (i, group), ax in zip(df_destination_age_male.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="proportion_%", title=str(i),ax=ax ,kind='line',color='b',label='male' )
        ax.set_ylim([0, 6])
        ax.set_ylabel('percentage')
    plt.tight_layout()

    #female in red
    for (i, group), ax in zip(df_destination_age_female.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="proportion_%", title=str(i),ax=ax ,kind='line',color='r',label='female')
        ax.set_ylim([0, 6])
    plt.tight_layout()
    plt.show()
    
# Display number of travelers number per country given age and sex
# @arg(in) df_destination_age_male : DataFrame  of males  
# @arg(in) df_destination_age_female : DataFrame  of females  
def travellerNumberCountryPlot(df_destination_age_male,df_destination_age_female):
    SIZE = 10
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    #male in blue
    fig, axes = plt.subplots(nrows=5, ncols=2)
    fig.set_size_inches(10, 12, forward=True)
    
    for (i, group), ax in zip(df_destination_age_male.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="population_in_thousands", title=str(i),ax=ax ,kind='line',color='b',label='male' )
        ax.set_ylim([200, 6000])
        ax.set_ylabel('people in thousands')
    plt.tight_layout()


    #female in red
    for (i, group), ax in zip(df_destination_age_female.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="population_in_thousands", title=str(i),ax=ax ,kind='line',color='r',label='female') 
        ax.set_ylim([200, 12000])
        ax.set_ylabel('people in thousands')
    plt.tight_layout()
    plt.show()
    
# Display number of travelers number per country
# @arg(in) df_destination_total : DataFrame of all travellers 
def destinationTotalPlot(df_destination_total):
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    ax = df_destination_total.sort_values('population_in_thousands',ascending=False).plot(x='country_destination', y='population_in_thousands',kind='bar', figsize=(20,8))
    ax.set_ylabel('people in thousands')
    ax.set_title('Destination of travelers')
    plt.show()

    
# export a list of id with the invalid ages to .csv file
# @arg(in) df : DataFrame
def exportInvalidAge(df):
    #invalid age
    df_invalid_age = df.loc[(df['age']==-1), ['id']]
    df = df[df['age'] != -1]

    #not specified age
    df_invalid_age= pd.concat([df_invalid_age, (df[df['age'].isnull()])])
    df.dropna(subset=['age'],inplace = True)

    #export
    pd.DataFrame(df_invalid_age, columns=list(df_invalid_age.columns)).to_csv('invalid_age_user_id.csv', index=False, encoding="utf-8") 
    print('file saved')
    return df
    
# plot age
# @arg(in) df : DataFrame    
def plotAge(df):
   
    df2 = df[df['age'] != -1]
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    df2.id.groupby(df2.age).count().plot(kind='bar', alpha=0.6, color='b',figsize=(20,8))
 
    ax = plt.axes()

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::2])
    ax.xaxis.set_ticklabels(ticklabels[::2])
    plt.show()
    
    
# plot gender
# @arg(in) df : DataFrame    
def plotGender(df):
    SIZE = 10
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    df.id.groupby(df.gender).count().plot(kind='bar', alpha=0.4, color='b',figsize=(10,3))
    plt.ylabel('Number of  users')
    plt.show()
    
# clean gender
# @arg(in) df : DataFrame
def cleanGender(df):
    #Assign unknown to category '-unknown-' and '-other-'
    df.loc[df['gender']=='-unknown-', 'gender'] = 'UNKNOWN'
    df.loc[df['gender']=='OTHER', 'gender'] = 'UNKNOWN'
    return df

# clean First_affiliate_tracked
# @arg(in) df : DataFrame
def cleanFirst_affiliate_tracked(df):
    df.loc[df['first_affiliate_tracked'].isnull(), 'first_affiliate_tracked'] = 'untracked'
    return df

# Clean Date_first_booking
def cleanDate_First_booking(df):
    df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])
    return df
    
def plotDate_First_booking_years(df):
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    df.date_first_booking.value_counts().plot(kind='line', linewidth=1,figsize=(20,8))
    plt.ylabel('Number of bookings')
    plt.xlabel('Time in days')
    plt.title('Number of bookings throughout time')
    plt.show()
    
def plotDate_First_booking_months(df):
    df.id.groupby([df.date_first_booking.dt.month]).count().plot(kind='bar',figsize=(20,8))
    plt.xticks(np.arange(12), calendar.month_name[1:13])
    plt.xlabel('Month')
    plt.ylabel('Number of bookings')
    plt.title('Number of bookings over the months of the year')
    plt.show()

def plotDate_First_booking_weekdays(df):
    df.id.groupby([df.date_first_booking.dt.weekday]).count().plot(kind='bar',figsize=(20,8))
    plt.xticks(np.arange(7), calendar.day_name[0:7])
    plt.xlabel('Week Day')
    plt.ylabel('Number of bookings')
    plt.title(s='Number of bookings per day in the week')
    plt.show()

    
#export file to csv
# @arg(in) filename : name of file in String, with .csv at the end 
# @arg(in) df : DataFrame
def saveFile(df, filename):
    pd.DataFrame(df, columns=list(df.columns)).to_csv(filename, index=False, encoding="utf-8") 
    print('file saved')    
    
# Delete NaN given subset
# @arg(in) df : DataFrame  
# @arg(in) subset :  name of the subset, between ''
def cleanSubset(df, subset):
    df2 = df.dropna(subset=[subset])
    removed = round(100-len(df2)/len(df)*100,2)
    print(removed, '% have been removed from the original dataframe')
    return df2


# Delete NaN given subset
# @arg(in) df : DataFrame  
# @arg(in) subset :  name of the subset, between ''
def cleanSubset(df, subset):
    df2 = df.dropna(subset=[subset])
    removed = round(100-len(df2)/len(df)*100,2)
    print(removed, '% have been removed from the original dataframe')
    return df2


# Plot histogram of a column
# @arg(in) df : column of a dataframe. ex : df['my_col']
def plotHist(df_col):
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE) 
    df_col.value_counts().plot(kind='bar', figsize=(20,8))


# Clean Action of file sessions.csv
# @arg(in) df : DataFrame  
def cleanAction(df):
    df2 = copy.deepcopy(df)
    df.loc[df['action'].isnull(), ['action']] = '-unknown-'
    df.loc[df['action_type'].isnull(), 'action_type'] = '-unknown-'
    df.loc[df['action_detail'].isnull(), 'action_detail'] = '-unknown-'
    removed = round(100-len(df)/len(df2)*100,2)
    print(removed, '% have been removed from the original dataframe')
    return df


# Create feature : total number of action per user file sessions.csv
# @arg(in) df : DataFrame
# @arg(out) data_session_number : DataFrame of user_id with their number of action  
def createActionFeature(df):
    data_session_number = df.groupby(['user_id'], as_index=False)['action'].count()
    return data_session_number


# Plot total number of action per user file sessions.csv
# @arg(in) df : DataFrame
def plotActionFeature(data_session_number):
    num_min= 1
    num_max= max(data_session_number['action'])
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE) 
    data_session_number.hist(bins = np.logspace(np.log10(num_min),np.log10(num_max),100), figsize=(20,8))
    plt.xlabel('Number of actions')
    plt.ylabel('Number of users')
    plt.gca().set_xscale("log")
    plt.title('Total number of actions per user')

    
# Create feature : average time spent per user file sessions.csv
# @arg(in) df : DataFrame
# @arg(out) data_time_mean : DataFrame of user_id with their average time spent     
def createAverageTimeFeature(df):
    data_time_mean = df.groupby(['user_id'], as_index=False).mean()
    return data_time_mean


# Create feature : total time spent per user file sessions.csv
# @arg(in) df : DataFrame
# @arg(out) data_time_total : DataFrame of user_id with their total time spent     
def createTotalTimeFeature(df):
    data_time_total = df.groupby(['user_id'], as_index=False).sum()
    return data_time_total


# Plot time spent sessions.csv
# @arg(in) df : DataFrame
def plotTimeFeature(data_time, type_plot):

    # Showing users spending at least 20 seconds by session on average
    time_min= 20
    time_max= max(data_time)
    
    plt.figure(figsize=(20,8))
    SIZE = 20
    plt.rc('font', size=SIZE)                # controls default text sizes
    plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)          # legend fontsize
    plt.rc('figure', titlesize=SIZE)
    plt.hist(data_time, bins =np.logspace(np.log10(time_min),np.log10(time_max),10000), log=True)
    plt.xlabel('Time in seconds')
    plt.ylabel('Number of users')
    plt.gca().set_xscale("log")
    
    if type_plot == 'total': 
        plt.title('Total time spent in second per user')
    elif type_plot == 'mean': 
        plt.title('Average time spent in second per user')
    elif type_plot == 'dist': 
        plt.title('Time spent in second per session')