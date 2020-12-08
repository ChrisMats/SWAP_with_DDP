import wandb
from .bases import *
        
class Trainer(BaseTrainer):
    def __init__(self, wraped_defs):
        super().__init__()

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
        
        self.visible_world = self.model.visible_world
        self.base_id = torch.cuda.current_device()
        
        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)        
        self.report_intermediate_steps = True     
        self.best_model = deepcopy(self.org_model_state)
        
    def train(self):
        self.test_mode = False
        self.load_session(self.restore_only_model)
        self.print_train_init()
        
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        epoch_bar = tqdm(epoch_bar, desc='Epoch', leave=False)
        for self.epoch in epoch_bar:
            iter_bar = enumerate(self.trainloader)
            iter_bar = tqdm(iter_bar, desc='Training', leave=False, total=len(self.trainloader))
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, it=it)   
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0: 
                        self.epoch_step()
                
        print(" ==> Training done")
        self.evaluate()
        self.save_session(verbose=True)
        
    def global_step(self, **kwargs):
        self.model.train() 
        self.optimizer.zero_grad()
        
        images, labels = kwargs['batch']
        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True)        
        outputs = self.model(images)
        
        loss = self.criterion(outputs, labels)            
        loss.backward() 
        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
        self.optimizer.step()  

        if self.scheduler_type == 'OneCycleLR':
            self.scheduler.step()        
    
        if self.iters % self.log_every == 0 or self.iters == 1:
            self.logging({'train_loss': loss.item(),
                         'learning_rate': self.get_lr()})    
    
    def epoch_step(self, **kwargs):          
        self.evaluate()
        self.save_session()        
        
        if self.scheduler is not None:       
            if self.scheduler_type == 'MultiStepLR':
                self.scheduler.step()
            if self.scheduler_type == 'ReduceLROnPlateau':
                if self.scheduler.mode == 'min':
                    self.scheduler.step(self.val_loss)
                else:
                    self.scheduler.step(self.val_acc)
                
    def evaluate(self, dataloader=None, report_cm=False, **kwargs):
        self.model.eval()
        if dataloader == None:
            dataloader=self.valloader
            
        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return
        val_loss = edict()
        n_classes = self.model.n_classes
        metric = self.metric_fn(self.model.n_classes, dataloader.dataset.int_to_labels, mode="val")
        iter_bar = tqdm(dataloader, desc='Validating', leave=False, total=len(dataloader))
        
        val_loss = []
        with torch.no_grad():
            for images, labels in iter_bar:                
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                outputs = self.model(images)                               
                loss = self.criterion(outputs, labels)
                
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)
        
        self.val_loss = np.array(val_loss).mean()        
        eval_metrics = metric.get_value()
        self.val_acc = eval_metrics.val_accuracy
        
                
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
        metric = self.metric_fn(self.model.n_classes, dataloader.dataset.int_to_labels, mode="test")
        iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))
        
        with torch.no_grad():
            for images, labels in iter_bar: 
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                outputs = self.model(images)                               
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