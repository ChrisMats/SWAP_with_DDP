from utils import *

from torchvision.transforms import *
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

from torch import nn
from torchvision import models
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, OneCycleLR
    
class BaseSet(Dataset):
    
    def attr_from_dict(self, param_dict):
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])

    def get_trans_list(self, transform_dict):
        transform_list = []   
            
        if transform_dict['VerticalFlip']:
            transform_list.append(RandomVerticalFlip(p=0.5))

        if transform_dict['HorizontalFlip']:
            transform_list.append(RandomHorizontalFlip(p=0.5))
            
        if transform_dict['RandomRotatons']['apply']:
            transform_list.append(RandomRotation(degrees=transform_dict['RandomRotatons']['angle']))  
            
        if transform_dict['Resize']['apply']:
            transform_list.append(Resize((transform_dict['Resize']['height'],
                                         transform_dict['Resize']['width'])))
            
        if transform_dict['RandomAffine']['apply']:
            temp_d = transform_dict['RandomAffine']
            transform_list.append(RandomAffine(degrees=temp_d['degrees'],
                                              translate=temp_d['translate'], 
                                              scale=temp_d['scale'], 
                                              shear=temp_d['shear']))               
            
        if transform_dict['CenterCrop']['apply']:
            transform_list.append(CenterCrop((transform_dict['CenterCrop']['height'],
                                             transform_dict['CenterCrop']['width'])))
            
        if transform_dict['RandomCrop']['apply']:
            padding = transform_dict['RandomCrop']['padding']
            transform_list.append(RandomCrop((transform_dict['RandomCrop']['height'],
                                         transform_dict['RandomCrop']['width']),
                                        padding=padding if padding > 0 else None))     
            
        if transform_dict['RandomPerspective']['apply']:
            (RandomPerspective(transform_dict['RandomPerspective']['distortion_scale']))
            
        if transform_dict['ColorJitter']['apply']:
            temp_d = transform_dict['ColorJitter']
            transform_list.append(ColorJitter(brightness=temp_d['brightness'],
                                              contrast=temp_d['contrast'], 
                                              saturation=temp_d['saturation'], 
                                              hue=temp_d['hue'])) 
            
        transform_list.append(ToTensor())
        if transform_dict['Normalize']:
            transform_list.append(Normalize(mean=self.mean, 
                                            std=self.std)) 
        if transform_dict['RandomErasing']['apply']:
            temp_d = transform_dict['RandomErasing']
            transform_list.append(RandomErasing(scale=temp_d['scale'],
                                              ratio=temp_d['ratio'], 
                                              value=temp_d['value']))             
        
        return transform_list


    def get_transforms(self):
        
        if self.mode == 'train':
            aplied_transforms = self.train_transforms
        if self.mode in ['val', 'eval']:
            aplied_transforms = self.val_transforms
        if self.mode == 'test':
            aplied_transforms = self.test_transforms
        
        transformations = self.get_trans_list(aplied_transforms)
        transforms = Compose(transformations)

        return transforms
    
    def remove_norm_transform(self):
        no_norm_transforms = deepcopy(self.transform.transforms)
        no_norm_transforms = [trans for trans in no_norm_transforms 
                              if not isinstance(trans, Normalize)]
        self.transform = Compose(no_norm_transforms)
    
    def Unormalize_image(self, image):
        norm = [trans for trans in self.transform.transforms 
                if isinstance(trans, Normalize)][0]
        unorm_mean = tuple(- np.array(norm.mean) / np.array(norm.std))
        unorm_std = tuple( 1.0 / np.array(norm.std))
        return Normalize(unorm_mean, unorm_std)(image)
    
    @staticmethod    
    def get_validation_ids(total_size, val_size, json_path, dataset_name, seed_n=42, overwrite=False):
        """ Gets the total size of the dataset, and the validation size (as int or float [0,1] 
        as well as a json path to save the validation ids and it 
        returns: the train / validation split)"""
        idxs = list(range(total_size))   
        if val_size < 1:
            val_size = int(total_size * val_size)    

        if not os.path.isfile(json_path) or overwrite:
            print("Creating a new train/val split for \"{}\" !".format(dataset_name))
            random.Random(seed_n).shuffle(idxs)
            train_split = idxs[val_size:]
            val_split = idxs[:val_size]
            save_json(val_split, json_path)    

        else:
            val_split = load_json(json_path)
            if val_size != len(val_split):
                print("Found updated validation size for \"{}\" !".format(dataset_name))
                _, val_split = BaseSet.get_validation_ids(total_size, val_size, json_path, 
                                                          dataset_name, seed_n=42, overwrite=True)
            train_split = list(set(idxs) - set(val_split))

        return train_split, val_split         
    
    
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()  
        self.base_id = torch.cuda.current_device()
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def get_out_channels(self, m):
        def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
        c=children(m)
        if len(c)==0: return None
        for l in reversed(c):
            if hasattr(l, 'num_features'): return l.num_features
            res = self.get_out_channels(l)
            if res is not None: return res
            
    def get_submodel(self, m, min_layer=None, max_layer=None):
        return list(m.children())[min_layer:max_layer]
    
    def freeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d):
                layer.eval()
                
    def unfreeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d):
                layer.train()
                
    def freeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = False
            
    def unfreeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = True

    def initialize_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()  

    def freeze_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.eval()  
                
    def unfreeze_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.train()                  
                
    def init_weights(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data)

    def print_trainable_params(self, submodel=None):
        submodel = self if submodel is None else submodel
        for name, param in submodel.named_parameters():
            if param.requires_grad:
                print(name)   
                
    @property
    def visible_world(self):
        return torch.cuda.device_count()   
   
    @property
    def visible_ids(sefl):
        return list(range(torch.cuda.device_count()))
    
    @property
    def device_id(self):    
        did = torch.cuda.current_device() 
        assert self.base_id == did
        return did              
    
    @property
    def is_rank0(self):
        return is_rank0(self.device_id)
   
                
class BaseTrainer:
    def __init__(self):
        super().__init__()  
        self.val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.val_acc = 0.
        self.best_val_acc = 0.
        self.iters = 0
        self.epoch0 = 0
        self.epoch = 0
        self.base_id = torch.cuda.current_device()        
        
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def reset(self):
        if is_parallel(self.model):
            self.model.module.load_state_dict(self.org_model_state)
            self.model.module.to(self.model.module.device_id)            
        else:
            self.model.load_state_dict(self.org_model_state)
            self.model.to(self.model.device_id)
        self.optimizer.load_state_dict(self.org_optimizer_state)
        print(" Model and optimizer are restored to their initial states ")
        
    def load_session(self, restore_only_model=False, model_path=None):
        self.get_saved_model_path(model_path=model_path)
        if os.path.isfile(self.model_path) and self.restore_session:        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            if is_parallel(self.model):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device_id)
            self.org_model_state = model_to_CPU_state(self.model)
            self.best_model = deepcopy(self.org_model_state)
            if restore_only_model:
                return
            
            self.iters = checkpoint['iters']
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.model_path, checkpoint['epoch']))

        elif not os.path.isfile(self.model_path) and self.restore_session:
            print("=> no checkpoint found at '{}'".format(self.model_path))
    
    def get_saved_model_path(self, model_path=None):
        if model_path is None:
            model_saver_dir = os.path.join(os.getcwd(), 'checkpoints')
            check_dir(model_saver_dir)
            self.model_path = os.path.join(model_saver_dir, self.model_name)            
        else:
            self.model_path = os.path.abspath(model_path)
        
    def save_session(self, model_path=None, verbose=False):
        if self.is_rank0:
            self.get_saved_model_path(model_path=model_path)
            if verbose:
                print("Saving model as {}".format(self.model_name) )
            state = {'iters': self.iters, 'state_dict': self.best_model,
                     'optimizer': opimizer_to_CPU_state(self.optimizer), 'epoch': self.epoch,
                    'parameters' : self.parameters}
            torch.save(state, self.model_path)
        synchronize()

    def save_parallel_models(self, model_path=None, verbose=False, include_epochs=False):
        self.get_saved_model_path(model_path=model_path)
        # I don't understand the udnerlying stuff, so I am just hacking things in
        if include_epochs:
            self.model_path = self.model_path + '_epoch{}'.format(self.epoch)
        rank = self.device_id
        self.model_path = self.model_path + '_rank{}'.format(rank)
        if verbose:
            print("Saving model as {}".format(self.model_name) )
        
        # couldn't make it work with the best_model so I hacked it
        synchronize()
        state = {'iters': self.iters, 'state_dict': self.best_model,
                 'optimizer': opimizer_to_CPU_state(self.optimizer), 'epoch': self.epoch,
                'parameters' : self.parameters}
        torch.save(state, self.model_path)
        synchronize()
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def print_train_init(self):
        if self.is_rank0: 
            print("Start training with learning rate: {}".format(self.get_lr()))    
            
    def logging(self, logging_dict):
        if not self.is_rank0: return
        wandb.log(logging_dict, step=self.iters)    
             
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