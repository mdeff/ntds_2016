# A Network Tour of Data Science

This repository contains the exercises for the EPFL master course
[A Network Tour of Data Science][epfl]. There is two types of exercises.

[epfl]: http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558

The **Data Scientist toolkit**, a set of [tools][toolkit], mostly in Python, to
help during the Data Science process.

1. [Introduction][00_intro].
2. Data acquisition & exploration: [demo][01_demo], [exercise][01_ex], [solution][01_sol].
3. Data exploitation: demo, exercise.
4. High Performance Computing: demo, exercise.
5. Cloud: demo, exercise.
6. Graph tools: demo, exercise.

[toolkit]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/toolkit
[00_intro]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/00_introduction.ipynb
[01_demo]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_demo_acquisition_exploration.ipynb
[01_ex]:      http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_ex_acquisition_exploration.ipynb
[01_sol]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_sol_acquisition_exploration.ipynb

**Machine Learning** (ML) & **Graph Signal Processing** (GSP) [algorithms].
These exercises are designed so as to familiarize yourself with the algorithms
presented in class.

1. Graph Science: [exercise][01_ex].
2. Clustering: [exercise][02_ex], [assignment][02_ass].
3. Classification: exercise.
4. Neural Networks: exercise.
5. Signal Processing on Graphs: exercise.
6. Sketching and other randomized approaches: exercise.

[algorithms]: http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/algorithms
[01_ex]:      http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/01_ex_graph_science.ipynb
[02_ex]:      http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ex_clustering.ipynb
[02_ass]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ass_clustering.ipynb

## Installation

1. Install Python.
	* Windows: we recommend to install [Anaconda]. Please install version 3.5.
	  Most of the packages we'll use during the exercises are included in the
	  distribution. An other option is the [Windows Subsystem for Linux][wsl],
	  available on Windows 10, which allows you to install packages as if you
	  were on Ubuntu.
	* Mac: we recommend that you use the [Homebrew] package manager and install
	  Python with `brew install python3`. You can also use [Anaconda].
	* Linux: please use your package manager to install the latest Python 3.x.

2. Clone the course repository. You may need to first install [git].
   ```sh
   git clone https://github.com/mdeff/ntds_2016.git
   cd ntds_2016
   ```

3. Optionally, create a [virtual environment][venv]. 
   ```sh
   pyvenv /path/to/new/virtual/env
   . /path/to/new/virtual/env/bin/activate
   ```
   > A virtual environment allows you to install a different set of packages for
   > each of your Python project. Each project thus stays cleanly separated from
   > each other. It is a good practice but by no means necessary. You can read
   > more about virtual environments on this [blog post][venv_blog]. Anaconda
   > users, see [here][conda_venv].

4. Install the packages we'll use from [PyPI], the Python Package Index.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

   * If it fails, it is probably because you need to install some native
	 packages with your package manager. Please read the error messages and
	 remember, Google is your friend !

   * Depending on your installation, `pip` may refer to Python 2 (you can
	 verify with `pip -V`). In that case, use `pip3` instead of `pip`.

   * Anaconda users can also install packages with `conda install packname`.
	 See [here][conda_install] for your options.

5. Verify that you have a working installation by running a simple test.
   Again, you may need to call `python3`.
   ```sh
   python check_install.py  # or make test
   ```

   * If you are on Windows with Anaconda and get
	 `WARNING (theano.configdefaults): g++ not detected!`, you may want to
	 install [mingw-w64](http://mingw-w64.org) with `conda install mingw
	 libpython`. Otherwise your Deep Learning models will run extremly slowly.
	 This may however not work for Python 3.5, see this [GitHub
	 issue][theano_windows_py35] for a workaround.

6. Open the jupyter web interface and play with the notebooks !
   ```sh
   jupyter notebook
   ```

[Homebrew]: http://brew.sh
[wsl]: https://msdn.microsoft.com/en-us/commandline/wsl/about
[Anaconda]: https://www.continuum.io/downloads#windows
[conda_install]: http://stackoverflow.com/a/18640601/3734066
[conda_venv]: http://conda.pydata.org/docs/using/envs.html
[venv]: https://docs.python.org/3/library/venv.html
[venv_blog]: https://realpython.com/blog/python/python-virtual-environments-a-primer/
[PyPI]: https://pypi.python.org
[git]: https://git-scm.com/downloads
[theano_windows_py35]: https://github.com/Theano/Theano/issues/3376#issuecomment-235034897

## License

All codes and examples are released under the terms of the [MIT
License](LICENSE.txt).
