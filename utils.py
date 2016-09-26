import pandas as pd
import urllib.request
import os.path


def get_data(directory):
    """Download dataset and split it among many files."""

    if not os.path.exists(directory):
        os.makedirs(directory)

        filename = os.path.join(directory, 'original.xls')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
        url += '00350/default%20of%20credit%20card%20clients.xls'
        urllib.request.urlretrieve(url, filename)

        data = pd.read_excel(filename, skiprows=[0], index_col=0)

        col_csv = ['LIMIT', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
        col_xls = ['DELAY1', 'DELAY2', 'DELAY3', 'DELAY4', 'DELAY5', 'DELAY6']
        col_hdf = ['BILL1', 'BILL2', 'BILL3', 'BILL4', 'BILL5', 'BILL6']
        col_sql = ['PAY1', 'PAY2', 'PAY3', 'PAY4', 'PAY5', 'PAY6']
        col_json = ['DEFAULT']

        data.columns = col_csv + col_xls + col_hdf + col_sql + col_json

        filename_csv = os.path.join(directory, 'demographics.csv')
        filename_xls = os.path.join(directory, 'delays.xls')
        filename_hdf = os.path.join(directory, 'bills.hdf5')
        filename_sql = os.path.join(directory, 'payments.sqlite')
        filename_json = os.path.join(directory, 'target.json')

        data.loc[:, col_csv].to_csv(filename_csv)
        data.loc[:, col_xls].to_excel(filename_xls)
        data.loc[:, col_hdf].to_hdf(filename_hdf, 'bills')
        data.loc[:, col_sql].to_sql('payments', 'sqlite:///' + filename_sql)
        data[col_json].to_json(filename_json)
