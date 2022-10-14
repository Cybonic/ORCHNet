#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/


'''

Version: 3.1 
 - pretrained model is automatically loaded based on the model and session names 
 
'''

import argparse
import yaml
from shutil import copyfile
import os
import shutil
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import signal, sys
from sklearn.neighbors import NearestNeighbors
from torch import optim
import torch 

from networks.AttDLNet import *


from datetime import datetime

import random
from torch.utils.data import DataLoader, random_split
from utils.utils import dump_info
from dataloader.ORCHARDS import ORCHARDS
from trainer import Trainer
from networks import model
from utils.utils import generate_descriptors
from utils.retrieval import place_knn,evaluation,comp_gt_table,retrieval_metric,relocal_metric
from utils.viz import myplot

def comp_eval_idx(pred,gt):
    '''
    
    
    '''
    tp,fp,tn,fn = [],[],[],[]
    for p,g in zip(pred,gt): 
        p = np.array(p).tolist()
        n_true_pos = len(g)
        n_pos_hat = len(p)
        ttp = []
        ffp = []
        if n_true_pos > 0: # postive 
            # Loops exist, we want to know if it can retireve the correct frame
            ttp = [c for c in p if c in g]
            if len(tp)== 0:
               ffp = p
            
        else: # Negative
            # Loop does not exist: we want to know if it can retrieve a frame 
            # with a similarity > thresh
            if n_pos_hat > 0:
                ffp = p

        tp.append(ttp)
        fp.append(ffp)

    return np.array(tp),np.array(fp)


def color_retrieval_on_map(num_samples,anchors,tp,fp):
    #num_samples = pose.shape[0]
    # print(tp)
    c = np.array(['gainsboro']*num_samples)
    c[:anchors] = 'k'
    c[anchors] = 'y'
    s = np.ones(num_samples)*15

    if len(tp)>0:
        c[tp] = 'g'
        s[tp] = 200
    if len(fp)>0:
        c[fp] = 'r'
        s[fp] = 50
    
    
    s[anchors]= 100
    

    return(c,s)

class relocalization():
    def __init__(self,loader,model, device = 'cpu', descriptor_path=None):

        self.loader   = loader
        self.model    = model.to(device)
        self.pose   = loader.dataset.pose
        self.anchor = loader.dataset.anchor
        self.all_idx = loader.dataset.idx_universe
        self.device = device
        self.path_to_descriptors = descriptor_path
    
    def _get_top_cand(self,pred_idx,pred_scores,pos_thrs=0.5,top=1):
        top_pred_cand_idx = pred_idx[:,:top]
        top_pred_sores = pred_scores[:,:top]
        top_cand_hat = []

        for c,s in zip(top_pred_cand_idx,top_pred_sores):
            c_hat = c[(s<pos_thrs).any() and (s>0).any()]
            top_cand_hat.append(c_hat[0] if len(c_hat)>0 else [])
        return(top_cand_hat)


    def relocalize(self,sim_thres=0.5, burn_in=10, range_thres=1,top_cand=1):

        self.gt_table = comp_gt_table(self.pose,self.anchor,range_thres)

        self.model.eval()
        self.true_loop_idx = np.array([np.where(self.gt_table[i]==1)[0] for i in range(self.gt_table.shape[0])])
        
        if self.path_to_descriptors == None or not os.path.isfile(self.path_to_descriptors):
            descriptors = generate_descriptors(self.model,self.loader,self.device)
        else:
            descriptors = torch.load(self.path_to_descriptors)

        max_top = np.max(top_cand)
        self.pred_idx, self.pred_scores  = place_knn(descriptors,
                                                    top_cand = max_top, 
                                                    burn_in  = burn_in, 
                                                    sim_thres = sim_thres,
                                                    )

        overall_scores = {}

        for top in top_cand:
            top_cand_hat = self._get_top_cand(self.pred_idx,self.pred_scores,pos_thrs=sim_thres,top=top)
            scores = relocal_metric(top_cand_hat,self.true_loop_idx)
            overall_scores[top] = scores
            print(scores)
        
        return(overall_scores)

    
    def plot(self, sim_thrs = 0.5, record_gif=False, top=25, name= 'relocalization.gif'):

        top_cand_hat = self._get_top_cand(self.pred_idx,self.pred_scores,pos_thrs=sim_thrs,top=top)
        tp_idx,fp_idx = comp_eval_idx(top_cand_hat,self.true_loop_idx)

        plot = myplot(delay = 0.001)
        if record_gif == True:
            plot.record_gif(name)

        plot.init_plot(self.pose[:,0],self.pose[:,1],c='whitesmoke',s=10)
        plot.xlabel('m')
        plot.ylabel('m')

        num_samples = self.pose.shape[0]
        #self.pred_idx
        all_idx = np.arange(num_samples)[::-1]
        for i in range(1,num_samples,10):
            # Idx manipulation to improve plotting
            idx = np.arange(i+1)
            idx = np.concatenate(( np.setxor1d(all_idx,idx)[::-1],idx))

            c,s = color_retrieval_on_map(self.pose.shape[0],i,tp_idx[i],fp_idx[i])
            plot.update_plot(self.pose[idx,0],self.pose[idx,1],color = c[idx] , offset= 1, zoom=-1,scale=s[idx])



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='AttVLAD_resnet50',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--cfg', '-f',
      type=str,
      required=False,
      default='sensor-cfg',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--resume', '-p',
      type=str,
      required=False,
      default='checkpoints/bev-rerecord_sparce-AttVLAD_resnet50-0.70.pth',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--memory',
      type=str,
      required=False,
      default='Disk',
      choices=['Disk','RAM'],
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--debug', '-b',
      type=bool,
      required=False,
      default=False,
      help='Directory to get the trained model.'
  )


  parser.add_argument(
      '--modality',
      type=str,
      required=False,
      default='bev',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--dataset',
      type=str,
      required=False,
      default='orchard-uk',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--device',
      type=str,
      required=False,
      default='cuda',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--descriptors',
      type=str,
      required=False,
      default = 'orchard-uk-rerecord_sparce-range-AttVLAD_resnet50',
      #default='rerecord_sparce-range-VLAD_resnet50_0.1@1',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--sequence',
      type=str,
      required=False,
      default='rerecord_sparce',
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()



  # open arch config file
  cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
  try:
    print("Opening data config file: %s" % cfg_file)
    sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  session_cfg_file = os.path.join('sessions', FLAGS.dataset + '.yaml')
  try:
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()


    print("----------")
  print("INTERFACE:")
  print("Root: ", SESSION['root'])
  print("Dataset: ", FLAGS.dataset)
  print("Sequence: ", FLAGS.sequence)
  print("Memory: ", FLAGS.memory)
  print("Model:  ", FLAGS.model)
  print("Debug:  ", FLAGS.debug)
  print("Resume: ", FLAGS.resume)
  print(f'Device: {FLAGS.device}')
  #print(f'batch size: {FLAGS.batch_size}')
  print(f'Debug: {FLAGS.debug}')
  print("----------\n")

  SESSION['val_loader']['data']['modality'] = FLAGS.modality
  SESSION['val_loader']['data']['dataset'] = FLAGS.dataset
  SESSION['val_loader']['data']['sequence'] = FLAGS.sequence
  SESSION['val_loader']['batch_size'] = 1
  
  dataset = SESSION['val_loader']['data']['dataset']

  descriptor_path = None
  if FLAGS.descriptors != None:
    descriptor_path = f'predictions/{dataset}/descriptors/{FLAGS.descriptors}.npy'

   ###################################################################### 
  orchard_loader = ORCHARDS(root          = SESSION['root'],
                            val_loader    = SESSION['val_loader'],
                            sensor        = sensor_cfg,
                            mode =  FLAGS.memory,
                            debug = FLAGS.debug)
                            
 
  modality = FLAGS.modality + '_param'

  model_ = model.ModelWrapper(FLAGS.model, loss=None , device= FLAGS.device,**SESSION[modality])
  checkpoint = torch.load(FLAGS.resume)
  model_.load_state_dict(checkpoint['state_dict'])
  
  run_name = {  'dataset': SESSION['train_loader']['data']['dataset'],
                'dataset': SESSION['train_loader']['data']['sequence'],
                'experiment':SESSION['experim_name'], 
                'model':SESSION['model']['type']
                }
 
 ###################################################################### 
  eval = relocalization(
          model  = model_,
          loader = orchard_loader.get_val_loader(),
          device = FLAGS.device,
          descriptor_path = descriptor_path

          )
  
  sim_thres = 0.3
  results = eval.relocalize(sim_thres = sim_thres,top_cand = list(range(1,25,1)),burn_in=1000)
  
  columns = ['top','recall','precision']
  rows = [[t,v['recall'],v['precision']] for t,v in results.items()]
  import pandas as pd

  top_cand = 5
  score =  round(results[top_cand]['recall'],3)

  df = pd.DataFrame(rows,columns = columns)
  df.to_csv(f'predictions/{dataset}/results/{FLAGS.sequence}-{FLAGS.modality}-{FLAGS.model}-{score}@{top_cand}.csv')
  
  gif_name = f'predictions/{dataset}/reloc/{FLAGS.descriptors}-{score}@{top_cand}.gif'

  eval.plot(sim_thrs = sim_thres, top = top_cand, record_gif=True, name = gif_name)


  
  