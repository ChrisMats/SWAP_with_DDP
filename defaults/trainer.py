import wandb
from .bases import *
import torch.distributed as dist
        
class Trainer(BaseTrainer):
    def __init__(self, wraped_defs):
        super().__init__()
        self.is_grid_search = False
        self.parameters = wraped_defs.parameters
        self.training_params = self.parameters.training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(wraped_defs.dataloaders)
        self.epoch_steps = len(self.trainloader)
        self.total_steps = int(len(self.trainloader) * self.epochs)
        
        self.model = wraped_defs.model
        self.criterion = wraped_defs.criterion        
        self.optimizer = wraped_defs.optimizer 
        self.scheduler = wraped_defs.scheduler
        self.scheduler_type = wraped_defs.scheduler_type
        self.metric_fn = wraped_defs.metric
                
        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)        
        self.report_intermediate_steps = True     
        self.best_model = deepcopy(self.org_model_state)


        # json this shit
        self.second_phase_start_epoch = 3
        
    def train(self):
        self.test_mode = False
        if not self.is_grid_search:
            self.load_session(self.restore_only_model)
        self.print_train_init()
        
        if is_parallel(self.model):
            n_classes = self.model.module.n_classes            
        else:
            n_classes = self.model.n_classes
        metric = self.metric_fn(n_classes, self.trainloader.dataset.int_to_labels, mode="train")
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
            
        for self.epoch in epoch_bar:
            self.model.train() 
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            
            if self.epoch == self.second_phase_start_epoch:
                print("Switching to the 2nd phase of training...")
                self.trainloader = self.nonddp_trainloader
                import copy
                self.model = copy.deepcopy(self.model.module)
                # I can't figure out how to re-init the optimizer using the wrapper thing
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3, weight_decay=1e-4)

            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, metric=metric, it=it)   
                
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0: 
                        self.epoch_step()  
                        self.model.train()         
                        
                synchronize()               
                        
            if self.scheduler_type in ['MultiStepLR', 'CosineAnnealingLR']:
                self.scheduler.step()                        
        if self.is_rank0:         
            print(" ==> Training done")
        if not self.is_grid_search:
            self.evaluate()
            self.save_session(verbose=True)
        
    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        
        metric = kwargs['metric']        
        images, labels = kwargs['batch']
        
        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True)        
        outputs = self.model(images)
        metric.add_preds(outputs, labels)
        
        loss = self.criterion(outputs, labels)            
        loss.backward() 
        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
        self.optimizer.step()  

        if self.scheduler_type == 'OneCycleLR':
            self.scheduler.step()        
    
        if not self.is_grid_search and self.is_rank0:
            if self.iters % self.log_every == 0 or self.iters == 1:
#                 Maybe an dist.all_reduce here to get the avg loss?
                self.logging({'train_loss': loss.item(),
                             'learning_rate': self.get_lr()})
                self.logging(metric.get_value())     
                metric.reset()
                
    
    def epoch_step(self, **kwargs): 
        self.evaluate()
        
        if not self.is_grid_search:
            if self.epoch<self.second_phase_start_epoch:
                # self.save_session()  
                # just for debugging, keep next line commented out (and unccomment the previous one)
                self.save_parallel_models(verbose=True, include_epochs=True)  
            else:
                self.save_parallel_models(verbose=True, include_epochs=True)     

            if self.scheduler_type == 'ReduceLROnPlateau':
                if self.scheduler.mode == 'min':
                    self.scheduler.step(self.val_loss)
                else:
                    self.scheduler.step(self.val_acc)
                
    def evaluate(self, dataloader=None, report_cm=False, **kwargs):
        if not self.is_rank0: return        
        self.model.eval()
        if dataloader == None:
            dataloader=self.valloader
            
        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return
        val_loss = edict()
        if is_parallel(self.model):
            n_classes = self.model.module.n_classes            
        else:
            n_classes = self.model.n_classes
        metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="val")
        iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        
        val_loss = []
        with torch.no_grad():
            for images, labels in iter_bar:                
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_parallel(self.model):
                    outputs = self.model.module(images) 
                else:
                    outputs = self.model(images)                     
                loss = self.criterion(outputs, labels)
#                 Maybe an dist.all_reduce here to get the avg loss?
#                 Maybe an dist.all_gather here to get all the predictions?
                        # Maybe later when get_value (?)
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)
        
        self.val_loss = np.array(val_loss).mean()        
        eval_metrics = metric.get_value()
        self.val_acc = eval_metrics.val_accuracy
        
        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
                self.logging({'val_loss': round(self.val_loss, 5)})
            if self.val_acc > self.best_val_acc: 
                self.best_val_acc = self.val_acc
                if self.save_best_model: 
                    self.best_model = model_to_CPU_state(self.model)
            if self.val_loss <= self.best_val_loss:
                self.best_val_loss = self.val_loss                    
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)                 
        self.model.train()
    
    def test(self, dataloader=None, **kwargs):
        if not self.is_rank0: return
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.load_session(self.restore_only_model)
        self.model.eval()
        if dataloader == None:
            dataloader=self.testloader  
            
        results_dir = os.path.join(os.getcwd(), 'results', self.model_name)
        metrics_path = os.path.join(results_dir, "metrics_results.json")
        check_dir(results_dir)     

        test_loss = []
        results = edict()
        if is_parallel(self.model):
            n_classes = self.model.module.n_classes            
        else:
            n_classes = self.model.n_classes        
        metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="test")
        iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))
        
        with torch.no_grad():
            for images, labels in iter_bar: 
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                outputs = self.model.module(images)                               
                loss = self.criterion(outputs, labels)
                
                test_loss.append(loss.item())
                metric.add_preds(outputs, labels)
        
        self.test_loss = np.array(test_loss).mean()
        test_metrics = metric.get_value()
        self.test_acc = test_metrics.test_accuracy
        test_metrics['test_loss'] = round(self.test_loss, 5)
        
        self.model.train()
        save_json(test_metrics, metrics_path)
        
        print('\n',"--"*5, "{} evaluated on the test set".format(self.model_name), "--"*5,'\n')
        pprint(test_metrics)
        print('\n',"--"*35, '\n')
        
    def lr_grid_search(self, min_pow=-5, max_pow=-1, resolution=20, n_epochs=5, 
                       random_lr=False, report_intermediate_steps=False, keep_schedule=False):
        res_dir = "grid_search_results"
        res_dir = os.path.join(os.getcwd(), res_dir)        
        self.is_grid_search = True
        self.save_best_model = False
        self.epochs = n_epochs
        if not keep_schedule:
            self.scheduler = None 
        pref_m = self.model_name
        self.model_name = 'grid_search'
        self.save_every = float("inf")   
        self.report_intermediate_steps = report_intermediate_steps
        if self.report_intermediate_steps:
            self.val_every = 1            
        else:
            self.log_every = float("inf")
            self.val_every = float("inf")
        
        v_losses = []
        v_accs = []
        if not random_lr:
            e = np.linspace(min_pow, max_pow, resolution)
            lr_points = 10**(e)
        else:
            lr_points = np.random.uniform(min_pow, max_pow, resolution)
            lr_points = 10**(e)
                    
        check_dir(res_dir)
        out_name = pref_m + "_grid_search_out.txt"
        out_name = os.path.join(res_dir, out_name)
        with open(out_name, "w") as text_file:
            print('learning rate \t val_loss \t val_AUC', file=text_file)
        for lr in tqdm(lr_points, desc='Grid search cycles', leave=False):
            if report_intermediate_steps:
                wandb.init(project=pref_m + '_grid_search', name=str(lr), reinit=True)     
        
            self.optimizer.param_groups[0]['lr'] = lr
            if not keep_schedule:
                self.scheduler_type = None
                self.scheduler = None       
            self.train()
            self.evaluate()
            v_losses.append(self.val_loss)
            v_accs.append(self.val_acc)
            if report_intermediate_steps:
                self.logging({'val_loss': self.val_loss,
                              'val_acc': self.val_acc})  
            with open(out_name, "a") as text_file:
                print('{} \t {} \t {}'.format(lr,self.val_loss,self.val_acc), file=text_file)
            self.reset()
            self.val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_acc = 0.
            self.best_val_acc = 0.
            self.iters = 0
            self.epoch0 = 0
            self.epoch = 0 
            if report_intermediate_steps:
                wandb.uninit()
            
        arg_best_acc = np.argmax(v_accs)
        best_acc = v_accs[arg_best_acc]
        best_lr_acc = lr_points[arg_best_acc]

        arg_best_vloss = np.argmin(v_losses)
        best_vloss = v_losses[arg_best_vloss]
        best_lr_vloss = lr_points[arg_best_vloss]

        print("The best val_acc is {} for lr = {}".format(best_acc, best_lr_acc))
        print("The best val_loss is {} for lr = {}".format(best_vloss, best_lr_vloss))
        
        fig, axs = plt.subplots(1,2, figsize=(15, 6))
        axs = axs.ravel()
        fig.suptitle('Grid search results')
        axs[0].plot(lr_points, v_losses)
        axs[0].scatter(best_lr_vloss, best_vloss, marker='*', c='r', s=100)
        axs[0].plot([best_lr_vloss]*2, [0, best_vloss], linestyle='--', c='r', alpha=0.5)
        axs[0].plot([lr_points[0], best_lr_vloss], [best_vloss]*2, linestyle='--', c='r', alpha=0.5)
        axs[0].set_xlabel('Learning rate')
        axs[0].set_ylabel('Validation loss')
        axs[0].set_xscale('log')
        axs[1].plot(lr_points, v_accs)
        axs[1].scatter(best_lr_acc, best_acc, marker='*', c='r', s=100)
        axs[1].plot([best_lr_acc]*2, [0, best_acc], linestyle='--', c='r', alpha=0.5)
        axs[1].plot([lr_points[0], best_lr_acc], [best_acc]*2, linestyle='--', c='r', alpha=0.5)
        axs[1].set_xlabel('Learning rate')
        axs[1].set_ylabel('Validation acc')
        axs[1].set_xscale('log')
        plt.savefig(os.path.join(res_dir, pref_m + '_grid_search_out.png'))   
        
        
   