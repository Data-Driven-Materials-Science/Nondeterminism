# Nondeterminism
### Associated Repository for "Identifying, Evaluating, and Addressing Nondeterminism in Mask R-CNNs"
The following lines of code are used to configure embedded randomness within the model structure 



#### Code required to disable embedded randomness within a model can be found in the training files within this repository or below
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
