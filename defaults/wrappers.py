from .models import *
from .datasets import *
from utils.helpfuns import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

class DefaultWrapper:
    """Class that wraps everything.

    Model, optimizers, schedulers, and dataloaders are initialized in this class.

    Attributes:
        param_attributes:
            All the fields in the .json file are stored as attributes here.
    """
    def __init__(self, parameters: EasyDict):
        """Inits the DefaultWrapper class.
        
        Args:
            parameters:
                Dictionary of paramaters read from a .json file.
        """
        super().__init__()
        parameters = EasyDict(deepcopy(parameters))
        self.param_attributes = list(parameters.keys())
        for key in parameters:
            setattr(self, key, parameters[key])        
        
    def instantiate(self):        
        """Initialize model, loss, metrics, dataloaders, optimizer and scheduler."""
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
        if self.training_params.second_phase_start_epoch < self.training_params.epochs:
            epochs = self.training_params.second_phase_start_epoch
        else:
            epochs = self.training_params.epochs
        scheduler_defs = self.init_scheduler(self.optimizer,
                                              self.optimization_params.default, 
                                              len(self.dataloaders.trainloader), 
                                              epochs)  
        self.attr_from_dict(scheduler_defs)
        
        # init loss functions
        self.criterion = self.init_criteria()  
        
        # init metric functions
        self.metric = DefaultClassificationMetrics
        
    
    def init_dataloaders(self, collate_fn=None) -> EasyDict:
        """Define dataset params and dataloaders.
        
        Args:
            collate_fn:
                Specific collate_fn for the torch.utils.data.DataLoader.
        
        Returns:
            A dict (EasyDict) with train, validation and test loaders. nonddp_trainloader is
            for the 2nd phase of SWAP training where we don't use the distributed sampler.
            
            {'trainloader': trainloader,
             'valloader': valloader,
             'testloader': testloader,
             'nonddp_trainloader':nonddp_trainloader}
        """ 
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

        return EasyDict({'trainloader': trainLoader,
                         'valloader' : valLoader,
                         'testloader' : testLoader,
                         'nonddp_trainloader': nonddp_trainloader,
                         })
        

    def init_model(self) -> Classifier:
        """Initialize the model.
        
        DDP broadcasts model states from rank 0 process to all other processes 
        in the DDP constructor, you don’t need to worry about different DDP processes 
        start from different model parameter initial values.   
        """
        model =  Classifier(self.model_params)
        model.to(self.device_id)
        if self.visible_world > 1 and torch.distributed.is_initialized():
            model = DDP(model, device_ids=[self.device_id])
        return model
    
    @staticmethod
    def init_optimizer(model, optimization_params:EasyDict) -> EasyDict:    
        """Initialize the optimizer.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.

        Returns:
            A dict (EasyDict) with optimizer and type keys.
            {'optimizer': optimizer (e.g. a torch.optim.Adam instance),
             'optimizer_type': optimizer_type (e.g. a string "Adam")}
        """
        optimizer_type = optimization_params.optimizer.type
        opt = optim.__dict__[optimizer_type]
        opt_params = optimization_params.optimizer.params
        optimizer = opt(filter(lambda p: p.requires_grad, model.parameters()), **opt_params)
        return EasyDict({"optimizer":optimizer, "optimizer_type":optimizer_type})
        
    @staticmethod        
    def init_scheduler(optimizer, optimization_params: EasyDict, steps_per_epoch: int=None, epochs: int=None) -> EasyDict:          
        """Initialize the learning rate scheduler.

        steps_per_epoch and epochs are set by the caller, they are not intended to be None.
        
        Args:
            optimization_params: EasyDict instance, read from the .json file.
        
        Returns:
            A dict (EasyDict) with scheduler and type keys.
            {'scheduler': scheduler (e.g. a torch.optim.lr_scheduler.OneCycleLR instance),
             'scheduler_type': scheduler_type (e.g. a string "OneCycleLR")}
        """
        scheduler_type = optimization_params.scheduler.type
        if scheduler_type not in optim.lr_scheduler.__dict__:
            return EasyDict({"scheduler":None, "scheduler_type":None})
        
        sch = optim.lr_scheduler.__dict__[scheduler_type]
        
        if sch.__name__ == 'OneCycleLR':
            max_lr = optimization_params.optimizer.params.lr
            sch_params = {"max_lr":max_lr, 
                          "steps_per_epoch":steps_per_epoch, 
                          "epochs":epochs,
                          "div_factor": max_lr/1e-8
                         }
            sch_params.update(optimization_params.scheduler.params.OneCycleLR)
        else:
            sch_params = optimization_params.scheduler.params[scheduler_type]
        scheduler = sch(optimizer, **sch_params) 
        return EasyDict({"scheduler":scheduler, "scheduler_type":scheduler_type})
    
    def init_criteria(self):          
        """Initialize the loss criteria.
        
        This is just an nn.CrossEntropy instance.
        """
        crit = nn.CrossEntropyLoss()  
        return crit
    
    def attr_from_dict(self, param_dict: EasyDict):
        """Function that makes the dictionary key-values into attributes.
        
        This allows us to use the dot syntax. Check the .json file for the entries.

        Args:
            param_dict: The dict we populate the class attributes from.
        """
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])    
        
    @property
    def parameters(self):
        return EasyDict({key : getattr(self, key) 
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
    
    