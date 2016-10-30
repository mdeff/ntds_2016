# A Network Tour of Data Science

This repository contains the exercises for the EPFL master course [A Network
Tour of Data Science][epfl] ([moodle]). There is two types of exercises.

[epfl]: http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558
[moodle]: http://moodle.epfl.ch/course/view.php?id=15299

The **Data Scientist toolkit**, a set of [tools][toolkit], mostly in Python, to
help during the Data Science process.

1. [Introduction][t00_intro].
2. Data acquisition & exploration: [demo][t01_demo], [exercise][t01_ex], [solution][t01_sol].
3. Data exploitation: [demo][t02_demo], [exercise][t02_ex].
4. High Performance Computing: demo, exercise.
5. Cloud: demo, exercise.
6. Graph tools: demo, exercise.

[toolkit]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/toolkit
[t00_intro]:  http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/00_introduction.ipynb
[t01_demo]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_demo_acquisition_exploration.ipynb
[t01_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_ex_acquisition_exploration.ipynb
[t01_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_sol_acquisition_exploration.ipynb
[t02_demo]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/02_demo_exploitation.ipynb
[t02_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/02_ex_exploitation.ipynb

**Machine Learning** (ML) & **Graph Signal Processing** (GSP) [algorithms].
These exercises are designed so as to familiarize yourself with the algorithms
presented in class.

1. Graph Science: [exercise][a01_ex], [solution][a01_sol].
2. Clustering: [exercise][a02_ex], [solution][a02_sol], [assignment][a02_ass].
3. Classification: [exercise][a03_ex], [exercise TensorFlow][a04_ex].
4. Neural Networks: exercise.
5. Signal Processing on Graphs: exercise.
6. Sketching and other randomized approaches: exercise.

[algorithms]: http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/algorithms
[a01_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/01_ex_graph_science.ipynb
[a01_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/01_sol_graph_science.ipynb
[a02_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ex_clustering.ipynb
[a02_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_sol_clustering.ipynb
[a02_ass]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ass_clustering.ipynb
[a03_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/03_ex_classification.ipynb
[a04_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/04_ex_tensorflow.ipynb

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

### Docker (TensorFlow on Windows)

[Docker](https://www.docker.com) is needed to install TensorFlow on Windows.
Docker is a virtualization method which helps to deploy applications inside
[containers]. You can think of them as lightweight virtual machines. [Install
docker][docker_win] first, then open a terminal and run the following command
to setup and run a tensorflow container:
```sh
docker run -it -p 8871:8888 -p 6011:6006 -v /path/to/exercises:/notebooks --name tf erroneousboat/tensorflow-python3-jupyter
```
You can now access the container's Jupyter notebook at <http://localhost:8871>.
Next time you can start the container with `docker start -i tf`.

[containers]: https://en.wikipedia.org/wiki/Operating-system-level_virtualization
[docker_win]: https://docs.docker.com/engine/installation/windows/

## License

All codes and examples are released under the terms of the [MIT
License](LICENSE.txt).
