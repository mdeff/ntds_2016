# A Network Tour of Data Science

This repository contains the exercises for the EPFL master course
[A Network Tour of Data Science][epfl]. There is two types of exercises.

[epfl]: http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558

The **Data Science toolkit**, a set of [tools][toolkit] available to the Data
Scientist.

1. [Introduction][00_intro].
2. Data acquisition & exploration ([demo][01_demo], exercise). 
3. Data exploitation (demo, exercise).
4. High Performance Computing (demo, exercise).
5. Cloud (demo, exercise).
6. Graph tools (demo, exercise).

**Machine Learning** (ML) & **Graph Signal Processing** (GSP) [algorithms].
These exercises are designed so as to familiarize yourself with the algorithms
presented in class.

1. Graph Science (exercise).
2. Clustering (exercise).
3. Classification (exercise).
4. Neural Networks (exercise).
5. Signal Processing on Graphs (exercise).
6. Sketching and other randomized approaches (exercise).

[algorithms]: http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/algorithms
[toolkit]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/toolkit
[00_intro]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/00_introduction.ipynb
[01_demo]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_demo_acquisition_exploration.ipynb

## Installation

1. Install Python.
	* Windows: we recommend to install [Anaconda]. Please install version 3.5.
	  Most of the packages we'll use during the exercises are included in the
	  distribution.
	* Mac: we recommend that you use the [Homebrew] package manager and install
	  Python with `brew install python3`.
	* Linux: please use your package manager to install the latest Python 3.x.

2. Create a [virtual environment][venv] (optional).
   ```sh
   pyvenv /path/to/new/virtual/env
   . /path/to/new/virtual/env/bin/activate
   ```

3. Clone the course repository.
   ```sh
   git clone https://github.com/mdeff/ntds_2016.git
   cd ntds_2016
   ```

4. Install the basic packages from [PyPI] (non-Anaconda users). If it fails, it
   is probably because you need to install some native packages with your
   package manager. Please read the error messages. Remember, Google is your
   friend !
   ```sh
   pip3 install -r requirements.txt  # make install
   ```

5. Verify that you have a working installation by running a simple test.
   ```sh
   python3 check_install.py  # make test
   ```

6. Open the jupyter web interface and play with the notebooks !
   ```sh
   jupyter notebook
   ```

[Homebrew]: http://brew.sh
[Anaconda]: https://www.continuum.io/downloads#windows
[venv]: https://docs.python.org/3/library/venv.html
[PyPI]: https://pypi.python.org
