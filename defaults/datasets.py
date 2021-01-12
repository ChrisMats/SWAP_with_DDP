from .bases import *
from torchvision.datasets import CIFAR10, ImageNet


class Cifar10(BaseSet, CIFAR10):
    """Dataset class loading CIFAR10.
    
    Uses the transforms read from the .json and applies to CIFAR10.
    """   

    n_classes = 10    
    img_channels = 3
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)    
    def __init__(self, dataset_params, mode='train'):        
        self.mode = mode
        self.attr_from_dict(dataset_params) 
        self.name = self.__class__.__name__
        root_dir = os.path.join(self.data_location, self.name)
                
        super().__init__(root=root_dir, train=True if self.mode != 'test' else False, 
                         transform=self.get_transforms(), download=self.download_data)   
        
        # just get the label/ int dictionary
        self.labels_to_int = self.class_to_idx
        self.int_to_labels = {val:key for key,val in self.labels_to_int.items()}     
        
        # split train / val
        if self.mode != 'test':
            val_id_json = os.path.join(root_dir, 'val_ids.json')
            train_ids, val_ids = self.get_validation_ids(total_size=len(self.data), 
                                                         val_size=self.validation_size, 
                                                         json_path=val_id_json, 
                                                         dataset_name=self.name)
            if self.mode == 'train':
                used_ids = train_ids
            elif self.mode in ['val', 'eval']:
                used_ids = val_ids
            else:
                raise ValueError("\"{}\" is not a valid mode. Please select one from [train, val, test]")

            self.data = self.data[used_ids]
            self.targets = [self.targets[idx] for idx in used_ids]    


class Imagenet(BaseSet, ImageNet):
    """Dataset class loading ImageNet.
    
    Uses the transforms read from the .json and applies to ImageNet.
    """   
    
    n_classes = 1000   
    img_channels = 3
    is_multiclass = True    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    def __init__(self, dataset_params, mode='train'):
        
        self.mode = mode
        self.attr_from_dict(dataset_params) 
        root_dir = self.data_location
        super().__init__(root=root_dir, split='train' if mode == 'train' else 'val', 
                         transform=self.get_transforms())        
        
        self.labels_to_int = self.class_to_idx
        self.int_to_labels = {val:key for key,val in self.labels_to_int.items()}   
        # I am using the validation set as a val/test set for now
        
    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample[isnan(sample)] = 0.
        return sample, target            
    
    