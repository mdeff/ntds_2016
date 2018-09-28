FROM python:3.5

MAINTAINER Michaël Defferrard <michael.defferrard@epfl.ch>

# Now that we have wheels, llvmlite includes llvm and numpy includes openblas.
# RUN echo "deb http://apt.llvm.org/jessie/ llvm-toolchain-jessie-3.9 main" >> /etc/apt/sources.list && \
#     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         llvm-3.9-runtime llvm-3.9-dev \
#         libatlas-base-dev liblapack-dev gfortran \
#         && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
# Alternative to ATLAS: libopenblas-dev
# ENV LLVM_CONFIG=llvm-config-3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends gfortran && \
    apt-get clean

WORKDIR /data
RUN git clone --depth=1 https://github.com/mdeff/ntds_2016.git repo && \
    mkdir mount

# Installing numpy first because of pip dependancy resolution bug.
RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install numpy && \
    pip --no-cache-dir install -r repo/requirements.txt && \
    jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
    jupyter nbextension install --py --sys-prefix vega && \
    jupyter nbextension enable --py --sys-prefix vega && \
    make -C repo test

# Add Tini.
ADD https://github.com/krallin/tini/releases/download/v0.13.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", \
     "--config=/data/repo/jupyter_notebook_config.py", "--allow-root"]
# Authentication: password and SSL certificate in config.

EXPOSE 8888
