from .helpfuns import *
import torch.distributed as dist

def compute_stats(dataloader):
    from tqdm import tqdm
    channels = dataloader.dataset[0][0].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x[0].mean([0,2,3]).cpu().numpy()
        x2_tot += (x[0]**2).mean([0,2,3]).cpu().numpy()

    channel_avr = x_tot/len(dataloader)
    channel_std = np.sqrt(x2_tot/len(dataloader) - channel_avr**2)
    return channel_avr,channel_std
        
def model_to_CPU_state(net):
    if is_parallel(net):
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.module.state_dict().items()}
    else:
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.state_dict().items()}
    return OrderedDict(state_dict)

def opimizer_to_CPU_state(opt):
    state_dict = {}
    state_dict['state'] = {}
    state_dict['param_groups'] = deepcopy(opt.state_dict()['param_groups'])

    for k, v in opt.state_dict()['state'].items():
        state_dict['state'][k] = {}
        if v:
            for _k, _v in v.items():
                if torch.is_tensor(_v):
                    elem = deepcopy(_v.cpu())
                else:
                    elem = deepcopy(_v)
                state_dict['state'][k][_k] = elem
    return state_dict     

class MovingMeans:
    def __init__(self, window=5):
        self.window = window
        self.values = []
        
    def add(self, val):
        self.values.append(val)
        
    def get_value(self):
        return np.convolve(np.array(self.values), np.ones((self.window,))/self.window, mode='valid')[-1]

def compute_stats(dataloader):
    from tqdm import tqdm
    channels = dataloader.dataset[0]['img'].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x['img'].mean([0,2,3]).cpu().numpy()
        x2_tot += (x['img']**2).mean([0,2,3]).cpu().numpy()

    channel_avr = x_tot/len(dataloader)
    channel_std = np.sqrt(x2_tot/len(dataloader) - channel_avr**2)
    return channel_avr,channel_std

def dist_average_tensor(tensor, mode='all', dst_rank=0):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor   
    world_size = float(dist.get_world_size())    
    if world_size < 2:
        return tensor     
    rt = tensor.clone()
    if mode == 'all':
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        dist.reduce(rt, dst=dst_rank, op=dist.ReduceOp.SUM)  
    rt /= world_size
    return rt

def dist_gather_tensor(tensor, mode='all', dst_rank=0, concatenate=True, cat_dim=0):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor 
    world_size = dist.get_world_size()
    if world_size < 2:
        return tensor     
    rt = tensor.clone()
    tensor_list = [torch.zeros_like(rt) for _ in range(world_size)]
    if mode == 'all':
        dist.all_gather(tensor_list, rt)
    else:
        if dist.get_backend() == 'nccl':
            raise RuntimeError("NCCL does not support gather. Please use all_gather mode=\"all\"")
        if dist.get_rank() == dst_rank:
            dist.gather(rt, tensor_list, dst=dst_rank)  
        else:
            dist.gather(rt, [], dst=dst_rank)  
    if concatenate:
        tensor_list = torch.cat(tensor_list, dim=cat_dim)
    
    return tensor_list

def dist_average_model_weights(model, mode='all'):
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return 
    world_size = float(dist.get_world_size())
    if world_size < 2:
        return     
    for param in model.parameters():
        if mode == 'all':
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        else:
            dist.reduce(param.data, dst=dst_rank, op=dist.ReduceOp.SUM)
        param.data /= world_size
        
