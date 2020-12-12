# SWAP_with_DDP
Repo for the ScaDaMaLe project


## TO DO:
- [x] Basic training for CIFAR10
- [ ] (Optional) Finetune model and hyperparameters
- [ ] Vanila DDP - Single server
    - [ ] Basic DDP handlers and launch defs
    - [ ] Divice_id everywhere 
    - [ ] Dataloaders
    - [ ] Training loop
    - [ ] Logging (vanila)    
    - [ ] Evaluation (vanila)        
    - [ ] Save / load

- [ ] Final DDP - multiple servers
- [ ] SWAP
- [ ] Fix readme 


## Stochastic Weight Averaging in Parallel (SWAP) in PyTorch
- [ ] Add project description, contributors etc
 
## Install dependencies etc.

- Python 3.8+ 
- Pytorch 1.7+

### Install using conda
- Using comands\
```conda create -n swap python=3.8 scikit-learn easydict matplotlib wandb tqdm -y```\
```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y```

- Using the .yml file\
```conda env create -f environment.yml```

### Install using docker (rootless)
- [ ] Add dockerfile



### Results
- [ ] Add main results
