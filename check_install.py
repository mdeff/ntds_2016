print('A Network Tour of Data Science: Python installation test')

import sys
major, minor = sys.version_info.major, sys.version_info.minor

if major is not 3:
    raise Exception('please use Python 3, you have Python {}.'.format(major))

try:
    import numpy
    import scipy
    import matplotlib
    import pandas
    import bokeh
    import jupyter
    import IPython, ipykernel
except:
    print('Your installation misses a package.')
    print('Please look for the package name below and install it with your '
          'package manager (conda, brew, apt-get, yum, pacman, etc.) or pip.')
    raise

print('You did successfully install Python {}.{} and '
      'the basic scientific Python packages.'.format(major, minor))
