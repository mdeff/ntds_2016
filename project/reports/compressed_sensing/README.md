# learn_cs_param

## Installation with docker
**TODO**

## Manual installation
**Warning:** TensorFlow does not support Windows yet.

  1. Install Python 3.x
    * Linux: Use your package manager to install Python 3.5 (or newer)
    * Mac: Either via the [Homebrew](http://brew.sh/) package manager using `brew install python3` or via [Anaconda](https://www.continuum.io/downloads)
    * Windows: Use [Anaconda](https://www.continuum.io/downloads)

  1. Clone the repository
    * ssh:

    ```bash
    git clone git@github.com:dperdios/learn_cs_param.git
    cd learn_cs_param
    ```

    * https:

    ```bash
    git clone https://github.com/dperdios/learn_cs_param.git
    cd learn_cs_param
    ```

  1. (optional) Create a dedicated environment. The code have been developed on Python 3.5
    * Using Anaconda:

    ```bash
    conda create -n learn_cs_param python=3.5
    ```

    * Using `pyenv`:

    ```bash
    pyvenv /path/to/new/virtual/env . /path/to/new/virtual/env/bin/activate
    ```

  1. Install the dependencies from `requirements.txt`. Please edit the TensorFlow version (CPU/GPU, Linux/OSX) you want to install. The code has been developed with TensorFlow 0.11.
    * Depending on your installation, `pip` may refer to Python 2 (you can verify with `pip -V`). In that case, use `pip3` instead of `pip`.

  ```bash
  pip install --upgrapde pip
  pip install -r requirements.txt
  ```

  1. **TODO** Check you have a working installation

  1. Launch Jupyter notebook.

  ```bash
  jupyter notebook
  ```

## License
**TODO**
