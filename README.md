# Nondeterminism

Note: Installation Guide fixed as of January 8th, 2021.

Allowing embedded randomness within a model strcuture and performing computations on a GPU instead of a CPU introduces a degree of nondeterminism into training results. The code within this repo provides examples of how to train Mask R-CNN Models consecutively, how to collect performance metrics to evaluate nondeterminism, and how to reduce or eliminate nondeterminism. 


This repository acts as a code base for the associated manuscript "Identifying, Evaluating, and Addressing Nondeterminism in Mask R-CNNs". (A link to this manuscript will be added when available) Here within, the code used and implemented, the training data, and results collected can be found. 

## Code
The required code to disable embedded randomness within a model can be found in the training files within this repository or below
 * Located at the beginning of training procedure
```python
random.seed(SEED)                               # Configures Python Random Library Seed
np.random.seed(SEED)                            # Configures NumPy Library Seed
torch.manual_seed(SEED)                         # Configures PyTorch Library Seed
torch.backends.cudnn.benchmark = False          # Disables Initial Benchmark Testing
torch.use_deterministic_algorithms(True)        # Chooses only determinalbe algorithms or throws and error
```
 * Located After Intializing Model Training Configurations (assuming model is named cfg)
```python
cfg.MODEL.DEVICE='cpu'                          # 'cpu' to force computions on CPU, 'cuda' to allow computations on a compatible gpu
cfg.INPUT.RANDOM_FLIP = "none"                  # TURNING OFF RANDOM DATA AUGMENTATION
cfg.SEED = SEED                                 # SETTING A SPECIFIC CNN SEED
```

## Annotations
All annotations used for training models were collected using VGG Image Annotator (VIA) and stored in JSON format. To generate new annotations, Via software is included in this repo under Data/Training_Data/VIA and title "via.html"

## Tools and Libraries
The code included implements the code or techniques from the following repositories:
 * https://github.com/rccohn/AMPIS
 * https://github.com/Data-Driven-Materials-Science/SALAS

