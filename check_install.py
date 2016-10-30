#!/bin/env python3

print('A Network Tour of Data Science: Python installation test')

import os
import sys
major, minor = sys.version_info.major, sys.version_info.minor

if major is not 3:
    raise Exception('please use Python 3, you have Python {}.'.format(major))

try:
    import numpy
    import scipy
    import matplotlib
    import sklearn

    import requests
    import facebook
    import tweepy

    import pandas
    import xlrd
    import xlwt
    import tables
    import sqlalchemy

    import statsmodels
    import sympy
    import autograd
    import bokeh
    import numba
    import Cython

    os.environ['KERAS_BACKEND'] = 'theano'  # Easier for Windows users.
    import keras
    import theano
    import tensorflow

    import jupyter
    import IPython, ipykernel

except:
    print('Your installation misses a package.')
    print('Please look for the package name below and install it with your '
          'package manager (conda, brew, apt-get, yum, pacman, etc.) or pip.')
    raise

print('You did successfully install Python {}.{} and '
      'most of the Python packages we will use.'.format(major, minor))
