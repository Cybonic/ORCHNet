
from cmath import nan
from base.base_trainer import BaseTrainer
from tqdm import tqdm  
import numpy as np
from utils.retrieval import evaluation,sim_knn,retrieval_knn
#from utils.viz import plot_retrieval_on_map
import torch
from utils.loss import L2_np 
from PIL import Image
import os

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
                        iter_per_epoch,
                        device = 'cpu',
                        run_name = 'default',
                        train_epoch_zero = False
                        ):

        super(Trainer, self).__init__(model, resume, config, iter_per_epoch,run_name=run_name,device=device,train_epoch_zero=train_epoch_zero)

        self.trainer_cfg    = config
        self.train_loader   = loader.get_train_loader()
        self.val_loader     = loader.get_val_loader()
        self.test_loader    = None
        #self.device         = device
        self.model          = model
        self.hyper_log      = config

        self.eval_metric = config['trainer']['eval_metric']
        self.top_cand_retrieval = config['retrieval']['top_cand']
        assert isinstance(self.top_cand_retrieval,list)

        self.train_metrics = None #StreamSegMetrics(len(labels))
        self.val_metrics = None #StreamSegMetrics(len(labels))
        self.batch_size = 1
        
        # Eval data
        try:
            self.map_idx = self.val_loader.dataset.get_map_idx()
            self.anchor_idx = self.val_loader.dataset.get_anchor_idx()
            self.gt_loops = self.val_loader.dataset.get_GT_Map()
            self.poses = self.val_loader.dataset.get_pose()
            self.comp_line_loop_table = self.val_loader.dataset.comp_line_loop_table
            #self.gt_line_loops = self.val_loader.dataset.gt_loop_table
            
        except: 
            self.map_idx = self.val_loader.dataset.dataset.get_map_idx()
            self.anchor_idx = self.val_loader.dataset.dataset.get_anchor_idx()
            self.gt_loops = self.val_loader.dataset.dataset.get_GT_Map()
            self.poses = self.val_loader.dataset.dataset.get_pose()
            self.comp_line_loop_table = self.val_loader.dataset.dataset.comp_line_loop_table
            self.gt_line_loops = self.val_loader.dataset.dataset.gt_line_loop_table

        #self.gt_line_loops = np.array([[value] for value in self.gt_line_loops ])

        self.gt_loops  = self.gt_loops[self.anchor_idx]
        self.true_loop = np.array([np.where(self.gt_loops[i]==1)[0] for i in range(self.gt_loops.shape[0])])

        

    def _reset_metrics(self):
        # Reset all evaluation metrics 
        #self.train_metrics.reset()
        #self.val_metrics.reset()
        pass 

    def _send_to_device(self,input):
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
        #value = value.cuda(non_blocking=True)
        return output

# ===================================================================================================================
# 
# ===================================================================================================================
    
    
    def _train_epoch(self, epoch):
        #self.html_results.save()
        
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
            
            #batch_data /= self.batch_size
            #batch_data.backward()

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
                #self.model.mean_grad()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
        #if epoch%50==0:
        #    self.batch_size += 50

        epoch_perfm = {}
        for key,value in epoch_loss_list.items():
            epoch_perfm[key] = np.mean(value)
        
        epoch_perfm['loss'] = epoch_loss/batch_idx
        #epoch_perfm ={'loss':,'ap':np.mean(epoch_ap),'an':np.mean(epoch_an)}
        self._write_scalars_tb('train',epoch_perfm,epoch)

        return epoch_perfm

            
# ===================================================================================================================
#    
# ===================================================================================================================

    def generate_descriptors(self,model,val_loader):
        dataloader = iter(self.val_loader)
        tbar = tqdm(range(len(self.val_loader)), ncols=100)

        self._reset_metrics()
        prediction_bag = {}
        idx_bag = []
        for batch_idx in tbar:
            input,inx = next(dataloader)
            #assert input.isnan().any() == False
            input = self._send_to_device(input)
            # Generate the Descriptor
            prediction = self.model(input)
            assert prediction.isnan().any() == False
            #print(torch.sum(torch.isnan(input)))
            # Keep descriptors
            for d,i in zip(prediction.detach().cpu().numpy().tolist(),inx.detach().cpu().numpy().tolist()):
                prediction_bag[int(i)] = d
        return(prediction_bag)
            
# ===================================================================================================================
#    
# ===================================================================================================================


    def _valid_epoch(self,epoch):
        

        sim_thres   = 0.5
        burn_in     = 60
        range_thres = 1
        top_cand = [1,5,25]

        # self.gt_table = comp_gt_table(self.pose,self.anchor,range_thres)
        from utils.relocalization import relocal_metric,sim_relocalize
        from eval_relocal import _get_top_cand
        
        self.model.eval()
        
        self.true_loop_idx = np.array([np.where(self.gt_loops[i]==1)[0] for i in range(self.gt_loops.shape[0])])
        
        descriptors = self.generate_descriptors(self.model,self.val_loader)
   

        max_top = np.max(top_cand)
        self.pred_idx, self.pred_scores  = sim_relocalize(  descriptors,
                                                            top_cand = max_top, 
                                                            burn_in  = burn_in, 
                                                            sim_thres = sim_thres,
                                                            )

        overall_scores = {}

        for top in top_cand:
            top_cand_hat = _get_top_cand(self.pred_idx,self.pred_scores,pos_thrs=sim_thres,top=top)
            scores = relocal_metric(top_cand_hat,self.true_loop_idx)
            overall_scores[top] = scores
            print(scores)
        
        return(overall_scores,descriptors)




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
                        
    
