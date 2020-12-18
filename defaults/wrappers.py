from .models import *
from .datasets import *
from utils.helpfuns import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

class DefaultWrapper:
    def __init__(self, parameters):
        super().__init__()
        parameters = edict(deepcopy(parameters))
        self.param_attributes = list(parameters.keys())
        for key in parameters:
            setattr(self, key, parameters[key])        
        
    def instantiate(self):        
        # init and get dataloaders
        if self.is_rank0:
            print("Initialising Dataloaders . . .")
        self.dataloaders = self.init_dataloaders()
        img_channels = self.dataloaders.trainloader.dataset.img_channels
        n_classes = self.dataloaders.trainloader.dataset.n_classes
        self.model_params.img_channels = img_channels
        self.model_params.n_classes = n_classes
        
        # init and get model
        if self.is_rank0:        
            print("Initialising Model . . .")        
        self.model = self.init_model()  
        
        if self.is_rank0:        
            print("Initialising Optimization methods . . ")                
        # init and get optimizer
        optimizer_defs = self.init_optimizer(self.model, self.optimization_params.default)  
        self.attr_from_dict(optimizer_defs)
        
        # init and get scheduler
        scheduler_defs = self.init_scheduler(self.optimizer,
                                              self.optimization_params.default, 
                                              len(self.dataloaders.trainloader), 
                                              self.training_params.epochs)  
        self.attr_from_dict(scheduler_defs)
        
        # init loss functions
        self.criterion = self.init_criteria()  
        
        # init metric functions
        self.metric = DefaultClassificationMetrics
        
    
    def init_dataloaders(self, collate_fn=None):
        # define dataset params and dataloaders  
        trainset = Cifar10(self.dataset_params, mode='train')
        valset = Cifar10(self.dataset_params, mode='eval')
        testset = Cifar10(self.dataset_params, mode='test')
        
        # distributed sampler 
        if self.visible_world > 1 and dist.is_initialized():        
            train_sampler = DS(trainset, num_replicas=self.visible_world, rank=self.device_id)
            self.dataloader_params['trainloader']['shuffle'] = False
            
        else:
            train_sampler = None

        # define distributed samplers etc
        trainLoader = DataLoader(trainset, **self.dataloader_params['trainloader'],sampler=train_sampler)
        self.dataloader_params['trainloader']['shuffle'] = True # Making sure that shuffling is ON again!
        nonddp_trainloader = DataLoader(trainset, **self.dataloader_params['trainloader'])
        valLoader = DataLoader(valset, **self.dataloader_params['valloader'])
        testLoader = DataLoader(testset, **self.dataloader_params['testloader'])
        
        if not len(valLoader) and self.is_rank0:
            warnings.warn("Warning... Using test set as validation set")
            valLoader = testLoader

        return edict({'trainloader': trainLoader, 
                       'valloader' : valLoader,
                       'testloader' : testLoader,
                       'nonddp_trainloader': nonddp_trainloader,
                       })
        

    def init_model(self):      
    # DDP broadcasts model states from rank 0 process to all other processes 
    # in the DDP constructor, you don’t need to worry about different DDP processes 
    # start from different model parameter initial values.
  
        model =  Classifier(self.model_params)
        model.to(self.device_id)
        if self.visible_world > 1 and torch.distributed.is_initialized():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    @staticmethod
    def init_optimizer(model, optimization_params):    
        # define optimizer
        optimizer_type = optimization_params.optimizer.type
        opt = optim.__dict__[optimizer_type]
        opt_params = optimization_params.optimizer.params
        optimizer = opt(filter(lambda p: p.requires_grad, model.parameters()), **opt_params)
        return edict({"optimizer":optimizer, "optimizer_type":optimizer_type})
        
    @staticmethod        
    def init_scheduler(optimizer, optimization_params, steps_per_epoch=None, epochs=None):          
        # define scheduler
        scheduler_type = optimization_params.scheduler.type
        if scheduler_type not in optim.lr_scheduler.__dict__:
            return edict({"scheduler":None, "scheduler_type":None})
        
        sch = optim.lr_scheduler.__dict__[scheduler_type]
        
        if sch.__name__ == 'OneCycleLR':
            max_lr = optimization_params.optimizer.params.lr
            sch_params = {"max_lr":max_lr, 
                          "steps_per_epoch":steps_per_epoch, 
                          "epochs":epochs,
                          "div_factor": max_lr/1e-8,
                         "final_div_factor": 1e-3}
        else:
            sch_params = optimization_params.scheduler.params[scheduler_type]
        scheduler = sch(optimizer, **sch_params) 
        return edict({"scheduler":scheduler, "scheduler_type":scheduler_type})
    
    def init_criteria(self):          
        # define criteria
        crit = nn.CrossEntropyLoss()  
        return crit
    
    def attr_from_dict(self, param_dict):
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])    
        
    @property
    def parameters(self):
        return edict({key : getattr(self, key) 
                      for key in self.param_attributes})
    
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(sefl):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        return torch.cuda.current_device() 
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)    
    
    