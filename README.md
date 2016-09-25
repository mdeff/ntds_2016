# A Network Tour of Data Science

This repository contains the exercises for the EPFL master course
[A Network Tour of Data Science](http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558).

## Installation

1. Install Python.
   * Windows: we recommend to install
	 [Anaconda](https://www.continuum.io/downloads#windows). Please install
	 version 3.5. Most of the packages we'll use during the exercises are
	 included in the distribution.
   * Mac: we recommend that you use the [Homebrew](http://brew.sh) package
     manager and install Python with `brew install python3`.
   * Linux: please use your package manager to install the latest Python 3.x.

2. Create a [virtual environment](https://docs.python.org/3/library/venv.html) (optional).
   ```
   pyvenv /path/to/new/virtual/env
   . /path/to/new/virtual/env/bin/activate
   ```

3. Clone the course repository.
   ```
   git clone https://github.com/mdeff/ntds_2016.git
   ```

4. Install the basic packages from [PyPI](https://pypi.python.org/)
   (non-Anaconda users). If it fails, it is probably because you need to
   install some native packages with your package manager. Please read the
   error messages.
   ```
   pip3 install -r requirements.txt
   ```

5. Verify that you have a working installation by runnning a simple test.
   ```
   python3 check_install.py
   ```
