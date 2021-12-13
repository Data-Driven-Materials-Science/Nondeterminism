# Satellite Detection Framework Installation Guide:
Note:
 * To properly install and run the code, there are a couple requirements. This code must be run on a native Linux or mac environment, there must be access to a GPU, and that GPU must not be an NVIDIA Tesla K40. There may be other GPUs that do not work, but that one definitively does not. To perform the installation, you must have access to a terminal AND have sudo permission however conda is a working alternative for not having sudo priviledges. Additionally, the example code is stored as a jupyter notebook. 
## First you want to install anaconda and git if you don’t have it
```bash
wget https://repo.continuum.io/archive/Anaconda3-2020.11-Linux-x86_64.sh
```
```bash
bash Anaconda3-2020.11-Linux-x86_64.sh
```
```bash
conda install -c anaconda git
```

## Clone git repository
```bash
git clone https://github.com/Data-Driven-Materials-Science/Nondeterminism
```
```bash
cd Nondeterminism
```

## Setup Virtual Environment
```bash
python3 -m venv nondeterminism_env
```
```bash
source nondeterminism_env/bin/activate
```

## Install Requirements File
```bash
pip install wheel
```
```bash
pip install -r requirements.txt
```

## Installing additional requirements not accounted for
```bash
[No longer recommending] sudo apt install build-essential
```
```bash
conda install gcc_linux-64 
```
```bash
[No longer recommending] sudo apt-get install python3.8-dev
```

## Installing COCO Python API
```bash
pip install pycocotools
```

### Sometimes, the above doesn’t work so if that is the case, try the following
```bash
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

## Installing PyTorch and TorchVision
```bash
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Sometimes, the above doesn’t work. If you recieve the error "[Errno 28] No space left on device", try the following
```bash
TMPDIR=/var/tmp pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Installing Requirements
```bash
python -m pip install -U scikit-image
```
```bash
pip install seaborn
```
```bash
pip install imantics 
```
```bash
python -m pip install -U scikit-image
```
## Installing Detectron2
```bash
python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
```

### Sometimes, the above doesn’t work so if that is the case, try the following
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
### If you experiece other issues, they can be solved [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues)
### Installing AMPIS
```bash
pip install -e .
```

### Open a python window in the terminal and try the following. If there are no errors, installation was completed properly
```bash
import pycocotools
import torch
import detectron2
import sat_helpers
```

