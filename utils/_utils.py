from .helpfuns import *

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
    if isinstance(net, torch.nn.DataParallel):
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
