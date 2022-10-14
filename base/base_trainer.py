import os, json, math, logging, sys, datetime
import numpy
import torch
from torch.utils import tensorboard
from utils import helpers
from utils import logger
#import utils.lr_scheduler
import torch.optim.lr_scheduler as lr_scheduler

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, resume, config, iters_per_epoch, train_logger=None, run_name='default',device='cpu'):
        self.model = model
        self.config = config

        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.start_epoch = 0
        self.improved = False
        self.best_log =None

        # SETTING THE DEVICE
        if device in ['gpu','cuda']:
            self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
            #self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
            self.model.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.do_validation = cfg_trainer['report_val']
        self.save_period = cfg_trainer['save_period']
        self.model_name = self.config['model']['type']
        
        # OPTIMIZER
        lr = float(config['optimizer']['args']['lr'])
        trainable_params = [#{'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()),'lr': 0.1*lr},
                                {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()),'lr': 0.1*lr},
                                 {'params':filter(lambda p:p.requires_grad, self.model.get_classifier_params()),'lr': lr}
                                ]

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        
        #model_params = sum([i.shape.numel() for i in list(model.parameters())])
        #opt_params   = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        #assert opt_params == model_params, 'some params are missing in the opt'
        
        # Learning Rate Sheduler
        # sheduler_type = config['optimizer']['lr_scheduler']['type']
        # t0 = config['optimizer']['lr_scheduler']['restart']
        
        # SHEDULER
        sheduler_type = config['optimizer']['lr_scheduler']
        args = config[sheduler_type]
        self.lr_scheduler = getattr(lr_scheduler,sheduler_type)(optimizer=self.optimizer,**args) 

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        checkpoint_dst_dir = os.sep.join([run_name['experiment'],run_name['dataset'],run_name['model']])
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], checkpoint_dst_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
       
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)
        
        if resume == None:
            pass
        elif resume  == 'best_model':
            resume = os.path.join(self.checkpoint_dir,'best_model.pth') 
            if not os.path.isfile(resume):
                resume = None
        elif resume  == 'auto':
            resume = os.path.join(self.checkpoint_dir,'checkpoint.pth') 
            if not os.path.isfile(resume):
                resume = None
        elif 'pth' in resume:
            self.logger.info(f'Loading from external weights')
        else:
            resume = None
            
        if resume: 
            self._resume_checkpoint(resume,name = self.model_name )
        
        
        self.dataset_name = run_name['dataset']
        writer_run_name = os.sep.join(  [run_name['experiment'],
                                        run_name['dataset'],
                                        run_name['model']
                                        ])

        if self.save_period<1: 
            # weights are  not saved
            date_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
            writer_run_name = os.path.join(writer_run_name,date_time)

        self.writer_dir = os.path.join(cfg_trainer['log_dir'], writer_run_name)
        self.writer = tensorboard.SummaryWriter(self.writer_dir)


    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        
        import GPUtil
        
        gpu = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
        print(torch.cuda.get_device_name())
        GPUtil.showUtilization()
        device = torch.device(f'cuda:{gpu[0]}' if n_gpu > 0 and len(gpu) >0 else 'cpu')
        
        print(f'GPU to be used: {gpu[0]}\n')
        
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    # =========================================================================================
    #
    # Train and Validation
    #
    # =========================================================================================

    def Train(self,train_entire_dataset=False):
        
        for epoch in range(self.start_epoch, self.epochs):
            
            # VAL EPOCH
            if (epoch % self.do_validation)  == 0:
                val_results,_ = self._valid_epoch(epoch)
                #val_results = {'recall':0, 'precision':0,'F1':0}
                self.logger.info('\n\n')
                
                for k, v in val_results.items():
                    self.logger.info(f'{str(k):15}: {v}')
                
                #self.print_table(val_results_label)
                log = {'epoch' : epoch, **val_results[1]} # Top 1 candidate
                
        
                # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if self.mnt_mode != 'off': # and epoch % self.config['trainer']['val_per_epochs'] == 0:
                    try:
                        if self.mnt_mode == 'min': self.improved = (log[self.mnt_metric] < self.mnt_best)
                        else: self.improved = (log[self.mnt_metric] > self.mnt_best)
                    except KeyError:
                        self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                        break
                        
                    if self.improved:
                        self.mnt_best = log[self.mnt_metric]
                        self.not_improved_count = 0
                        self.best_log = log
                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count > self.early_stoping:
                        self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                        self.logger.warning('Training Stoped')
                        break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0 and self.save_period > 0:
                self._save_checkpoint(epoch, save_best = self.improved)
                self.improved = False
            
            # TRAIN EPOCh
            train_results = self._train_epoch(epoch)
            
            self.lr_scheduler.step(train_results['loss'])

            if self.train_logger is not None:
                log = {'epoch' : epoch, **train_results}
                self.train_logger.add_entry(log)

        # Register best scores and hyper
        if self.best_log == None:
            self._write_hyper_tb(val_results)
        else:
            self._write_hyper_tb(self.best_log)
            
        # WHEN MONITORING IS OFF USE LAST RESULT
        #if self.mnt_mode != 'off' :
        #    score = self.mnt_best
        #else:
        #    score = val_results[self.mnt_metric]
        
        return(self.best_log)

    # =========================================================================================
    #
    # TEST 
    #
    # =========================================================================================

    def Eval(self):
        
        val_results,descriptors = self._valid_epoch(0)
        
        for k, v in val_results.items():
            self.logger.info(f'{str(k):15}: {v}')
        
        #score = val_results[self.mnt_metric]
        return(val_results,descriptors)

    
    def get_best_mnt_metric(self):
        return(self.mnt_best)


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch':self.model_name,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            #'monitor_best': self.mnt_best,
            'monitor_best': self.best_log,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")


    def _resume_checkpoint(self, resume_path,**argv):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        #self.start_epoch = checkpoint['epoch'] + 1
        try:
            self.not_improved_count = 0
            self.best_log = checkpoint['monitor_best']
            self.mnt_best = self.best_log[self.mnt_metric]
            weights_to_load = checkpoint['state_dict']
            self.model.load_state_dict(weights_to_load)
            self.logger.info(f'\nbest_score {self.mnt_metric}: %lf'%(self.mnt_best))
        except:
            self.logger.warning(f'This model was not trained by me\n')
        

            if 'deeplab' in argv['name'] and not 'best_model' in resume_path:
                weights_to_load = checkpoint['model_state']
                self.logger.info(f'best_score: %lf'%(checkpoint['best_score']))
                model_dict = self.model.model.state_dict()
                
                pretrained_dict = {}
                for k, v in weights_to_load.items():
                    if k in model_dict and v.size() == model_dict[k].size():
                        pretrained_dict[k]=v
                    else:
                        self.logger.info(f'Layer Dropted {k}')
        
                model_dict.update(pretrained_dict)
                self.model.model.load_state_dict(model_dict)

        if "logger" in checkpoint.keys():
            self.train_logger = checkpoint['logger']
        
        self.logger.info(f'\nCheckpoint <{resume_path}> (epoch {self.start_epoch}) was loaded, -> epochs {self.epochs}\n')
        

    def print_table(self,val_results_label):
        for key, value in val_results_label.items():
            if isinstance(value,dict):
                for key, value in value.items():
                    self.logger.info(f'{key:10}|{str(round(value,3)):5}|')
            self.logger.info(f'{key:10}|{str(round(value,3)):5}|')


    def _eval_metrics(self, output, target):
        raise NotImplementedError
