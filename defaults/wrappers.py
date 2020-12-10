from .models import *
from .datasets import *
from utils.helpfuns import *

class DefaultWrapper:
    def __init__(self, parameters):
        super().__init__()
        parameters = edict(deepcopy(parameters))
        self.param_attributes = list(parameters.keys())
        for key in parameters:
            setattr(self, key, parameters[key])        
        
    def instantiate(self):        
        # init and get dataloaders
        print("Initialising Dataloaders . . .")
        self.dataloaders = self.init_dataloaders()
        img_channels = self.dataloaders.trainloader.dataset.img_channels
        n_classes = self.dataloaders.trainloader.dataset.n_classes
        self.model_params.img_channels = img_channels
        self.model_params.n_classes = n_classes
        
        # init and get model
        print("Initialising Model . . .")        
        self.model = self.init_model()  
        
        print("Initialising Optimization methods . . ")                
        # init and get optimizer
        self.optimizer = self.init_optimizers()  
        
        # init and get scheduler
        self.scheduler = self.init_schedulers()   
        
        # init loss functions
        self.criterion = self.init_criteria()  
        
        # init metric functions
        self.metric = DefaultClassificationMetrics
        
    
    def init_dataloaders(self, collate_fn=None):
        # define dataset params and dataloaders  
        trainset = Cifar10(self.dataset_params, mode='train')
        valset = Cifar10(self.dataset_params, mode='eval')
        testset = Cifar10(self.dataset_params, mode='test')

        # define distributed samplers etc
        trainLoader = DataLoader(trainset, **self.dataloader_params['trainloader'])
        valLoader = DataLoader(valset, **self.dataloader_params['valloader'])
        testLoader = DataLoader(testset, **self.dataloader_params['testloader'])
        
        if not len(valLoader):
            warnings.warn("Warning... Using test set as validation set")
            valLoader = testLoader

        return edict({'trainloader': trainLoader, 
                       'valloader' : valLoader,
                       'testloader' : testLoader,})
        

    def init_model(self):        
        model =  Classifier(self.model_params)
        model.to(model.device_id)
        return model
    
    def init_optimizers(self):    
        # define optimizer
        self.optimizer_type = self.optimization_params.default.optimizer.type
        opt = optim.__dict__[self.optimizer_type]
        opt_params = self.optimization_params.default.optimizer.params
        return opt(filter(lambda p: p.requires_grad, self.model.parameters()), **opt_params)
        
    def init_schedulers(self):          
        # define scheduler
        self.scheduler_type = self.optimization_params.default.scheduler.type
        if self.scheduler_type not in optim.lr_scheduler.__dict__:
            self.scheduler_type = None
            return None
        
        sch = optim.lr_scheduler.__dict__[self.scheduler_type]
        
        if sch.__name__ == 'OneCycleLR':
            max_lr = self.optimization_params.default.optimizer.params.lr
            sch_params = {"max_lr":max_lr, 
                          "steps_per_epoch":len(self.dataloaders.trainloader), 
                          "epochs":self.training_params.epochs,
                          "div_factor": max_lr/1e-8,
                         "final_div_factor": 1e-3}
        else:
            sch_params = self.optimization_params.default.scheduler.params[self.scheduler_type]
        return sch(self.optimizer, **sch_params) 
    
    def init_criteria(self):          
        # define criteria
        crit = nn.CrossEntropyLoss()  
        return crit
        
    @property
    def parameters(self):
        return edict({key : getattr(self, key) 
                      for key in self.param_attributes})
    
    