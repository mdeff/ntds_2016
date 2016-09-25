import pandas as pd
import urllib.request
import os.path

def get_data(directory):
    """Download dataset and split it among many files."""

    if not os.path.exists(directory):
        os.makedirs(directory)

        filename = directory + 'original.xls'
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
        urllib.request.urlretrieve(url, filename)

        data = pd.read_excel(filename, skiprows=[0], index_col=0)

        columns_csv   = ['LIMIT', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
        columns_excel = ['DELAY1', 'DELAY2', 'DELAY3', 'DELAY4', 'DELAY5', 'DELAY6']
        columns_hdf   = ['BILL1', 'BILL2', 'BILL3', 'BILL4', 'BILL5', 'BILL6']
        columns_sql   = ['PAY1', 'PAY2', 'PAY3', 'PAY4', 'PAY5', 'PAY6']
        columns_json  = ['DEFAULT']

        data.columns = columns_csv + columns_excel + columns_hdf + columns_sql + columns_json

        data.loc[:, columns_csv].to_csv(directory + 'demographics.csv')
        data.loc[:, columns_excel].to_excel(directory + 'delays.xls')
        data.loc[:, columns_hdf].to_hdf(directory + 'bills.hdf5', 'bills')
        data.loc[:, columns_sql].to_sql('payments', 'sqlite:///' + directory + 'payments.sqlite')
        data[columns_json].to_json(directory + 'target.json')
