# Can't upgrade to 10.2 until PyTorch supports it (1.5?)
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# Grab tini so that Jupyter doesn't spray zombies everywhere
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /usr/bin/tini
RUN apt-get update && \
    # There's something weird going on with this particular image and fetching this particular library
    # Installing it separately first works; installing it implicitly with all the other deps below breaks
    # with a 400 Bad Request error. Might be transient.
    apt-get install -y --no-install-recommends \
        # Needed for conda
        curl ca-certificates bzip2 procps \ 
        # Needed for sanity
        neovim gdb wget man-db build-essential \
        # libzmq as installed by nodejs seems to work with clang when it doesn't work with gcc
        clang \
        # Needed for git installs
        git ssh-client \
        # Needed for nsight remotes
        openssh-server \
        # Needed for video output
        python-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev \
        libswresample-dev libavfilter-dev && \
    # Set up tini
    chmod +x /usr/bin/tini
ENV CC /usr/bin/clang

# Set up git
RUN git config --global user.name "Andrew Jones" && \
    git config --global user.email "andyjones.ed@gmail.com"


# Set up Miniconda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install things that tend to work better when installed from conda
RUN conda install -y numpy pytorch torchvision pandas -c pytorch && \
    conda clean -ya && \
    # Install Jupyter 7.5 because 7.6.1 has a bunch of lag with autoreload 
    pip install scipy numba psutil jupyter tqdm seaborn matplotlib wurlitzer ipython==7.5 flake8 sphinx && \
    # av>6.2 requires ffmpeg 4.0, which isn't available in ubuntu's default repos.
    pip install av==6.2

# Install ijavascript
RUN curl -sL https://deb.nodesource.com/setup_13.x | bash - && \
    apt-get install -y nodejs
ENV PATH $PATH:/usr/share/npm/node_modules/bin
RUN npm install -g --unsafe-perm ijavascript && \
    ijsinstall --install=global

# Install my jupyter workflow extension frontends
RUN pip install jupyter_contrib_nbextensions && \ 
    jupyter contrib nbextension install --user && \
    cd / && \
    git clone https://github.com/andyljones/noterminal && \
    cd noterminal && \
    jupyter nbextension install noterminal && \
    jupyter nbextension enable noterminal/main && \
    cd / && \
    git clone https://github.com/andyljones/stripcommon && \
    cd stripcommon && \
    jupyter nbextension install stripcommon && \
    jupyter nbextension enable stripcommon/main && \
    # This enables autoscroll, but I'm still unsure how to set the default line limit
    jupyter nbextension enable autoscroll/main 

# Install my workflow backends
RUN pip install git+https://github.com/andyljones/aljpy.git && \ 
    pip install git+https://github.com/andyljones/modulepickle && \
    pip install git+https://github.com/andyljones/snakeviz@custom && \
    pip install git+https://github.com/andyljones/noterminal && \
    pip install git+https://github.com/andyljones/pytorch_memlab && \
    rm -rf ~/.cache

# Copy the Jupyter config into place. 
ADD .jupyter /root/.jupyter
ADD .ipython /root/.ipython

# Set up the entrypoint script
ADD run.sh /usr/bin/

RUN mkdir -p /code
WORKDIR /code

ENTRYPOINT ["tini", "--", "run.sh"]
