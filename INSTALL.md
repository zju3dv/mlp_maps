### Set up the python environment
* NVIDIA GPU with CUDA 11.3 is required
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n mlp_maps python=3.8` to create a conda environment and activate it by `conda activate mlp_maps`)

* Python libraries
    * Install pytorch by `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113` 
    * Install torch-scatter by `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html`
    * Install kilonerf-cuda

        Option A: Install pre-compiled CUDA extension 

        Install pre-compiled CUDA extension  
        ```pip install lib/csrc/kilonerf_cuda/dist/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl```

        Option B: Build CUDA extension yourself

        Download magma from http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz then build and install to  `/usr/local/magma`
        ```
        sudo apt install gfortran libopenblas-dev
        sudo apt-get install freeglut3
        wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz
        tar -zxvf magma-2.5.4.tar.gz
        cd magma-2.5.4
        cp make.inc-examples/make.inc.openblas make.inc
        export GPU_TARGET="Maxwell Pascal Volta Turing Ampere"
        export CUDADIR=/usr/local/cuda
        export OPENBLASDIR="/usr"
        make
        sudo -E make install prefix=/usr/local/magma
        ```
        For further information on installing magma see: http://icl.cs.utk.edu/projectsfiles/magma/doxygen/installing.html

        Finally compile KiloNeRF's C++/CUDA code 
        ```
        cd lib/csrc/kilonerf_cuda
        python setup.py develop
        % Or use this command: TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" python setup.py develop
        ```
    * Install required packages by `pip install -r requirements.txt` 



### Set up datasets

#### ZJU-Mocap dataset

1. **Note that we refine the camera parameters of the ZJU-MoCap dataset.** If someone wants to download the refined ZJU-Mocap dataset, please fill in the [agreement](https://pengsida.net/project_page_assets/files/Refined_ZJU-MoCap_Agreement.pdf), and email Sida Peng (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.
2. Create a soft link:
    ```
    ROOT=/path/to/mlp_maps
    cd $ROOT/data
    ln -s /path/to/my_zjumocap my_zjumocap
    ```

#### NHR dataset

1. Download the NHR dataset at [here](https://wuminye.github.io/NHR/datasets.html) and process this data for our code. Or someone could download the processed data at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/El1HTEodvwhFmnCGo37e1gMBhho9Wh3SvjV5UWG_Z-t8Dw?e=KCpRhl). Note that both ways require to cite [the NHR paper](https://wuminye.github.io/NHR/datasets.html).
2. Create a soft link:
    ```
    ROOT=/path/to/mlp_maps
    cd $ROOT/data
    ln -s /path/to/nhr nhr
    ```
