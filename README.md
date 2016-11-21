# A Network Tour of Data Science

This repository contains the exercises for the EPFL master course [EE-558
A Network Tour of Data Science][epfl] ([moodle]). There is two types of
exercises.

[epfl]: http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558
[moodle]: http://moodle.epfl.ch/course/view.php?id=15299

The **Data Scientist toolkit**, a set of [tools][toolkit], mostly in Python, to
help during the Data Science process.

1. [Introduction][t00_intro].
2. Data acquisition & exploration: [demo][t01_demo], [exercise][t01_ex], [solution][t01_sol].
3. Data exploitation: [demo][t02_demo], [exercise][t02_ex], [solution][t02_sol].
4. High Performance Computing: [exercise][t03_ex], solution.
5. Data visualization: [exercise][t04_ex], solution.
6. Graph tools: demo, exercise.
7. Cloud: demo, exercise.

[toolkit]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/toolkit
[t00_intro]:  http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/00_introduction.ipynb
[t01_demo]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_demo_acquisition_exploration.ipynb
[t01_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_ex_acquisition_exploration.ipynb
[t01_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/01_sol_acquisition_exploration.ipynb
[t02_demo]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/02_demo_exploitation.ipynb
[t02_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/02_ex_exploitation.ipynb
[t02_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/02_sol_exploitation.ipynb
[t03_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/03_ex_hpc.ipynb
[t04_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/toolkit/04_ex_visualization.ipynb

**Machine Learning** (ML) & **Graph Signal Processing** (GSP) [algorithms].
These exercises are designed so as to familiarize yourself with the algorithms
presented in class.

1. Graph Science: [exercise][a01_ex], [solution][a01_sol].
2. Clustering: [exercise][a02_ex], [solution][a02_sol], [assignment][a02_ass], [solution][a02_sass].
3. Classification: [exercise][a03_ex], [exercise TensorFlow][a04_ex].
4. Neural Networks: [assignment][a05_ass], solution.
5. Recurrent Neural Networks: [assignment][a06_ass], solution.
5. Signal Processing on Graphs: assignment, solution.
6. Sketching and other randomized approaches: exercise.

[algorithms]: http://nbviewer.jupyter.org/github/mdeff/ntds_2016/tree/with_outputs/algorithms
[a01_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/01_ex_graph_science.ipynb
[a01_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/01_sol_graph_science.ipynb
[a02_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ex_clustering.ipynb
[a02_sol]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_sol_clustering.ipynb
[a02_ass]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_ass_clustering.ipynb
[a02_sass]:   http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/02_sol_assignment.ipynb
[a03_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/03_ex_classification.ipynb
[a04_ex]:     http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/04_ex_tensorflow.ipynb
[a05_ass]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/05_ass_convnet.ipynb
[a06_ass]:    http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/algorithms/06_ass_recurrent_nn.ipynb

The final evaluation is a **class project**, taking place at the end of the
semester. Read more about it in the [project description][desc].

[desc]: http://nbviewer.jupyter.org/github/mdeff/ntds_2016/blob/with_outputs/project/description.pdf

## Docker

The easiest way to play with the code is to run it inside a [docker] container,
a [lightweight virtualization method][virt].

[docker]: https://www.docker.com
[virt]: https://en.wikipedia.org/wiki/Operating-system-level_virtualization

1. [Install Docker][install] on your Windows, Mac or Linux machine.

2. Run the [image], which is automatically updated from this git repository.
   ```sh
   docker pull mdeff/ntds_2016  # to update it
   docker run --rm -i -p 8871:8888 -v ~/:/data/mount mdeff/ntds_2016
   ```

3. Access the container's Jupyter notebook at <http://localhost:8871>. Windows
   and Mac users may need to [redirect the port in VirtualBox][redirect]. There
   you'll find two folders:
   * `repo` contains a copy of this git repository. Nothing you modify in this
	 folder is persistent. If you want to keep your modifications, use `File`,
	 `Download as`, `Notebook` in the Jupyter interface.
   * `mount` contains a view of your home directory, from which you can
     persistently modify any of your files.

[install]: https://docs.docker.com/engine/installation/
[image]: https://hub.docker.com/r/mdeff/ntds_2016/
[redirect]: https://stackoverflow.com/a/33642903/3734066

### Container modification

If you want to use it for your projects and need additional software or Python
packages, you'll need to install them into the container.

1. Create your named container.
   ```sh
   docker run -i -p 8871:8888 -v ~/:/data/mount --name myproject mdeff/ntds_2016
   ```

2. Once you stop it, you'll be able to start it again with `docker start
   myproject`.

3. In another terminal, install packages while the container is running.
   ```sh
   docker exec -i myproject /bin/bash
   pip install mypackage
   apt-get install myotherpackage
   ```

## Manual installation

**Warning**: this may be problematic for Windows users, as TensorFlow does not
support Windows yet.

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
	 remember, Google is your friend ! You may look at the
	 [dockerfile](dockerfile) to get an idea of which setup is necessary on
	 a Debian / Ubuntu system.

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
