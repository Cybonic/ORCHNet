
from cmath import nan
from base.base_trainer import BaseTrainer
from tqdm import tqdm  
import numpy as np
from eval_knn import PlaceRecognition

# ===================================================================================================================
#       
#
#
# ===================================================================================================================

class Trainer(BaseTrainer):
    def __init__(self,  model,
                        resume,
                        config,
                        loader,
                        device = 'cpu',
                        run_name = 'default',
                        train_epoch_zero = True
                        ):

        super(Trainer, self).__init__(model, resume, config,run_name=run_name,device=device,train_epoch_zero=train_epoch_zero)

        self.trainer_cfg    = config
        self.train_loader   = loader.get_train_loader()
        self.val_loader     = loader.get_val_loader()
        self.test_loader    = None
        self.device         = device
        self.model          = model
        self.hyper_log      = config

        self.eval_metric = config['trainer']['eval_metric']
        self.top_cand_retrieval = config['retrieval']['top_cand']
        assert isinstance(self.top_cand_retrieval,list)

        self.train_metrics = None #StreamSegMetrics(len(labels))
        self.val_metrics = None #StreamSegMetrics(len(labels))
        self.batch_size = 1 # 
        
        window = 600 # Avoid the nearby frames

        self.eval_approach = PlaceRecognition(self.model ,self.val_loader,self.top_cand_retrieval,window,'L2',device)
     
        

    def _reset_metrics(self):
        # Reset all evaluation metrics 
        #self.train_metrics.reset()
        #self.val_metrics.reset()
        pass 

    def _send_to_device(self,input):
        # Send data structure to GPU 
        output = []
        if isinstance(input,list):
            for item in input:
                output_dict = {}
                for k,v in item.items():
                    value = v.to(self.device)
                    output_dict[k]=value
                output.append(output_dict)

        elif isinstance(input,dict):
            output = {}
            for k,v in input.items():
                value = v.to(self.device)
                output[k]=value
        else:
            output = input.to(self.device)
        return output

# ===================================================================================================================
# 
# ===================================================================================================================
    
    
    def _train_epoch(self, epoch):
        
        self.logger.info('\n')
        self.model.train()

        dataloader = iter(self.train_loader)
        tbar = tqdm(range(len(self.train_loader)), ncols=80)

        self._reset_metrics()
        epoch_loss_list = {}
        epoch_loss = 0
        
        self.optimizer.zero_grad()
        for batch_idx in tbar:
            
            input = next(dataloader)
            input_tonsor = self._send_to_device(input)
                    
            batch_data ,info= self.model(input_tonsor)
            

            for key,value in info.items():
                if key in epoch_loss_list:
                    epoch_loss_list[key].append(value.detach().cpu().item())
                else:
                    epoch_loss_list[key] = [value.detach().cpu().item()]
   
            # Accumulate error
            epoch_loss += batch_data.detach().cpu().item()

            tbar.set_description('T ({}) | Loss {:.10f}'.format(epoch,epoch_loss/(batch_idx+1)))
            tbar.update()

            if batch_idx % self.batch_size == 0:
                # Update model every batch_size iteration
                self.optimizer.step()
                self.optimizer.zero_grad()
            

        epoch_perfm = {}
        for key,value in epoch_loss_list.items():
            epoch_perfm[key] = np.mean(value)
        
        epoch_perfm['loss'] = epoch_loss/batch_idx
        self._write_scalars_tb('train',epoch_perfm,epoch)

        return epoch_perfm

            
# ===================================================================================================================
#    
# ===================================================================================================================



    def _valid_epoch(self,epoch):

        overall_scores = self.eval_approach.run()

        # Post on tensorboard
        for i, score in overall_scores.items():
            self._write_scalars_tb(f'val@{i}',score,epoch)
        return overall_scores,[]




# ===================================================================================================================
#    
# ===================================================================================================================
    def _write_scalars_tb(self,wrt_mode,logs,epoch):
        for k, v in logs.items():
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
            #if 'mIoU' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        if 'train' in wrt_mode:
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
    

    def _write_hyper_tb(self,logs):
        # https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3        
        hparam_dict = { "batch_size": self.hyper_log['train_loader']['batch_size'],
                        "experim_name": str(self.hyper_log['experim_name']),
                        "dataset": str(self.hyper_log['val_loader']['data']['dataset']),
                        "sequence": str(self.hyper_log['val_loader']['data']['sequence']),
                        "modality":self.hyper_log['val_loader']['data']['modality'],
                        "model": self.hyper_log['model']['type'],
                        "minibatch_size": self.hyper_log['model']['minibatch_size'],
                        "output_dim": self.hyper_log['model']['output_dim'],
                        "optim": str(self.hyper_log['optimizer']['type']),
                        "lr": self.hyper_log['optimizer']['args']['lr'],
                        "wd": self.hyper_log['optimizer']['args']['weight_decay'],
                        "lr_scheduler": self.hyper_log['optimizer']['lr_scheduler'],
        }

        metric_dict = logs
        self.writer.add_hparams(hparam_dict,metric_dict)
                        
    
