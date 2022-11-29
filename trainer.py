
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
                        ):

        super(Trainer, self).__init__(model, resume, config, iter_per_epoch,run_name=run_name,device=device)

        self.trainer_cfg    = config
        self.train_loader   = loader.get_train_loader()
        self.val_loader     = loader.get_val_loader()
        self.test_loader    = None
        #self.device         = device
        self.model          = model
        self.hyper_log      = config

        self.eval_metric = config['trainer']['eval_metric']
        self.top_cand_retrieval = config['retrieval']['top_cand']
        map_samples = len(self.val_loader.dataset.get_map_idx())
        self.top_cand_retrieval.append(round(map_samples/100))
        assert isinstance(self.top_cand_retrieval,list)

        self.train_metrics = None#StreamSegMetrics(len(labels))
        self.val_metrics = None #StreamSegMetrics(len(labels))
        self.batch_size = 10

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
        #self.model.model.requires_grad_(True)
        #self.train_loader.get_negatives_idx()
        dataloader = iter(self.train_loader)
        tbar = tqdm(range(len(self.train_loader)), ncols=80)

        self._reset_metrics()
        epoch_loss_list = {}
        epoch_an = []
        epoch_ap = []
        epoch_loss = 0
        self.batch_size = 10
        self.optimizer.zero_grad()
        for batch_idx in tbar:
            
            input = next(dataloader)
            input_tonsor = self._send_to_device(input)
            #pcl_pose = self._send_to_device(idx)            
            
            batch_data , info= self.model(input_tonsor)
            
            batch_data /= self.batch_size
            batch_data.backward()
            for key,value in info.items():
                if key in epoch_loss_list:
                    epoch_loss_list[key].append(value.detach().cpu().item())
                else:
                    epoch_loss_list[key] = [value.detach().cpu().item()]
            #epoch_an.append(batch_data['n'].detach().cpu().item())
            #epoch_ap.append(batch_data['p'].detach().cpu().item())
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

    def _valid_epoch(self, epoch):
        #self.html_results.save()
        self.logger.info('\n')
        self.model.eval()
        
        # Generate  Descriptors
        descriptors = self.generate_descriptors(self.model,self.val_loader)

        # If descriptors contain 'Nans' exit
        for  v,t in descriptors.items(): 
            if np.isnan(np.array(t)).any():
                return({'recall':-1, 'precision':-1,'F1':-1})

        #poses = self.val_loader.dataset.get_pose()
        map_idx = self.val_loader.dataset.get_map_idx()
        anchor_idx = self.val_loader.dataset.get_anchor_idx()
        
        gt_loops = self.val_loader.dataset.get_GT_Map()[anchor_idx]
        true_loop = np.array([np.where(gt_loops[i]==1)[0] for i in range(gt_loops.shape[0])])

        # Split Descriptors into Queries and Map
        query_dptrs = np.array([descriptors[i] for i in anchor_idx])
        map_dptrs = np.array([descriptors[i] for i in map_idx])
        # Retrieve loops 
        max_top = np.max(self.top_cand_retrieval)
        retrieved_loops ,scores = retrieval_knn(query_dptrs, map_dptrs, top_cand = max_top, metric = self.eval_metric)
        #retrieved_loops ,scores = euclidean_knnv2(query_dptrs,map_dptrs, top_cand= max_top)
        # retrieved_loops ,scores = sim_knn(query_dptrs,map_dptrs, top_cand = max_top,metric='cosine_loss')
        # Evaluate retrieval
        overall_scores = {}
        for top in self.top_cand_retrieval:
            scores = evaluation(retrieved_loops,true_loop, top = top)
            overall_scores[top]=scores
        # Post on tensorboard
        for i, score in overall_scores.items():
            self._write_scalars_tb(f'val@{i}',score,epoch)
        return overall_scores,descriptors

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
                        
    
