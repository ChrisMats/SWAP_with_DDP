# SWAP_with_DDP
Repo for the ScaDaMaLe project


## TO DO:
- [x] Basic training for CIFAR10 :heavy_check_mark:
- [ ] (Optional) Finetune model and hyperparameters
- [x] Vanila DDP - Single server :heavy_check_mark:
    - [x] Basic DDP handlers and launch defs
    - [x] Device_id everywhere 
    - [x] Dataloaders
    - [x] Training loop
    - [x] Logging (vanila)    
    - [x] Evaluation (vanila)        
    - [x] Save / load
    - [x] Fix issue that cause bottlenecks after each epoch (fixed: n_workers<2)

- [ ] Final DDP - multiple servers
    - [ ] (optional) work with SLURM
    - [ ] work on multiple servers  
    - [ ] DDP on log/eval etc     
- [ ] [SWAP](https://openreview.net/pdf?id=rygFWAEFwS)
- [ ] Fix readme 
- [ ] Clean up code 


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
