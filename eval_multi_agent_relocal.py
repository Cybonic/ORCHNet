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
from utils.relocalization import relocal_metric,comp_gt_table,sim_relocalize
from utils.viz import myplot
from dataloader.utils import load_dataset
from utils import loss as loss_lib

def retrieval_knn(query_dptrs,map_dptrs, top_cand,metric):
    
    #retrieved_loops ,scores = euclidean_knnv2(query_dptrs,map_dptrs, top_cand= max_top)
    metric_fun = loss_lib.get_distance_function(metric)
    scores,winner = [],[]

    for q in query_dptrs:
        q_torch,map_torch = loss_lib.totensorformat(q.reshape(1,-1),map_dptrs) 
        sim = metric_fun(q_torch,map_torch,dim=2).squeeze() # similarity-based metrics 0 :-> same; +inf: -> Dissimilar 
        sort_value,sort_idx = sim.sort() # Sort to get the most similar vectors first
        # save top candidates
        scores.append(sort_value.detach().cpu().numpy()[:top_cand])
        winner.append(sort_idx.detach().cpu().numpy()[:top_cand])

    return np.array(winner),np.array(scores)

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


def color_retrieval_on_map(num_samples,idx2plot,query,tp,fp):
    #num_samples = pose.shape[0]
    # print(tp)
    c = np.array(['gainsboro']*num_samples)
    c[idx2plot] = 'k'
    c[query] = 'y'
    s = np.ones(num_samples)*15

    #for tp,fp in zip(tp_vec,fp_vec):
    if len(tp)>0:
        c[tp] = 'g'
        s[tp] = 250
    if len(fp)>0:
        c[fp] = 'r'
        s[fp] = 100

    
    s[idx2plot]= 20
    

    return(c,s)

class MARelocalization():
    def __init__(self,loader,model, device = 'cpu', descriptor_path=None):

        self.loader   = loader
        self.model    = model.to(device)
        self.pose   = loader.dataset.get_pose()
        self.anchor_idx = loader.dataset.get_anchor_idx()
        self.database_idx = loader.dataset.get_map_idx()
        self.gt_loops = loader.dataset.get_GT_Map()
        #self.database_pose = loader.dataset.get_database_pose()
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

        # self.gt_table = comp_gt_table(self.pose,self.anchor,range_thres)

        self.model.eval()
        #self.anchor_loops, self.database_loops = np.where(self.gt_loop==1)
        self.database_loops = np.array([np.where(self.gt_loops[i]==1)[0] for i in range(self.gt_loops.shape[0])])
        
        if self.path_to_descriptors == None or not os.path.isfile(self.path_to_descriptors):
            descriptors = generate_descriptors(self.model,self.loader,self.device)
        else:
            descriptors = torch.load(self.path_to_descriptors)

        an_descriptor = np.array([descriptors[i] for i in self.anchor_idx])
        map_descriptor = np.array([descriptors[i] for i in self.database_idx])
       
        max_top = np.max(top_cand)
        self.pred_loop_idx, self.pred_scores  = retrieval_knn(an_descriptor,map_descriptor,max_top,'L2')

        overall_scores = {}

        # Evaluate retrieval
        from utils.metric import retrieve_eval

        overall_scores = {}
        for top in top_cand:
            scores = retrieve_eval(self.pred_loop_idx,self.database_loops, top = top)
            overall_scores[top]=scores
        # Post on tensorboard
        

        return(overall_scores)

    
    def plot(self, sim_thrs = 0.3, record_gif=False, top=25, name= 'relocalization.gif'):

        top_cand_hat = self._get_top_cand(self.pred_loop_idx,self.pred_scores,pos_thrs=sim_thrs,top=top)
        tp_idx,fp_idx = comp_eval_idx(top_cand_hat,self.database_loops)

        global_tp_idx = np.array([self.database_idx[i] for i in tp_idx])
        global_fp_idx = np.array([self.database_idx[i] for i in fp_idx])
        plot = myplot(delay = 0.001)
        if record_gif == True:
            plot.record_gif(name)

        plot.init_plot(self.pose[:,0],self.pose[:,1],c='whitesmoke',s=10)
        plot.xlabel('m')
        plot.ylabel('m')

        #num_samples = self.pose.shape[0]
        query_samples= self.anchor_idx.shape[0] 
        self.database_idx

        #self.pred_idx
        plot_idx = self.database_idx # start showing the database poses
        for i in range(1,query_samples,1):
            # Idx manipulation to improve plotting
            query_idx = self.anchor_idx[i] # Get current Query idx
            c,s = color_retrieval_on_map(self.pose.shape[0],plot_idx,query_idx,global_tp_idx[i],global_fp_idx[i]) # Colorize based on the results
            plot.update_plot(self.pose[:,0],self.pose[:,1],color = c, offset= 1, zoom=-1,scale=s)
            plot_idx = np.append(plot_idx,[query_idx]) # append the query IDX to be plot later



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='VLAD_pointnet',
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
      default='checkpoints/[00,02,05,06]_VLAD_pointnet.pth',
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
      default='pcl',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--session',
      type=str,
      required=False,
      default='fuberlin',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--device',
      type=str,
      required=False,
      default='cpu',
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--descriptors',
      type=str,
      required=False,
      default = 'place/descriptors/pcl-VLAD_pointnet_0.09@1',
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
  
  parser.add_argument(
      '--batch_size',
      type=int,
      required=False,
      default=10,
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()



  # open arch config file
  cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
  print("Opening data config file: %s" % cfg_file)
  sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))


  session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
  print("Opening session config file: %s" % session_cfg_file)
  SESSION = yaml.safe_load(open(session_cfg_file, 'r'))


  SESSION['model']['type'] = FLAGS.model
  print("----------")
  print("INTERFACE:")
  print("Root: ", SESSION['root'])
  print("Dataset: ", FLAGS.session)
  print("Sequence: ", FLAGS.sequence)
  print("Memory: ", FLAGS.memory)
  print("Model:  ", FLAGS.model)
  print("Debug:  ", FLAGS.debug)
  print("Resume: ", FLAGS.resume)
  print(f'Device: {FLAGS.device}')
  #print(f'batch size: {FLAGS.batch_size}')
  print(f'Debug: {FLAGS.debug}')
  print("----------\n")


  descriptor_path = None
  if FLAGS.descriptors != None:
    descriptor_path = f'predictions/{FLAGS.session}/{FLAGS.descriptors}.npy'
    assert os.path.isfile(descriptor_path),'Descriptor File does not exist!'

   ###################################################################### 
  loader, run_name = load_dataset(FLAGS,SESSION)

 
  modality = FLAGS.modality + '_param'
  model_ = model.ModelWrapper(**SESSION['model'],loss= [], **SESSION[modality])

  checkpoint = torch.load(FLAGS.resume,map_location=torch.device(FLAGS.device))
  
  try: 
    model_.load_state_dict(checkpoint['state_dict'])
  except:
    print("No Model Loaded!")
    exit()

  
  run_name = {  'dataset': SESSION['train_loader']['data']['dataset'],
                'dataset': SESSION['train_loader']['data']['sequence'],
                'experiment':SESSION['experim_name'], 
                'model':SESSION['model']['type']
                }
 
 ###################################################################### 
  eval = MARelocalization(
          model  = model_,
          loader = loader.get_val_loader(),
          device = FLAGS.device,
          descriptor_path = descriptor_path

          )
  
  sim_thres = 0.2
  results = eval.relocalize(sim_thres = sim_thres,top_cand = list(range(1,26,1)),burn_in=10)
  
  columns = ['top','recall','precision']
  rows = [[t,v['recall'],v['precision']] for t,v in results.items()]
  import pandas as pd

  top_cand = 1
  score =  round(results[top_cand]['recall'],3)

  df = pd.DataFrame(rows,columns = columns)

  results_dir = os.path.join('predictions',FLAGS.session,'reloc','results')
  if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

  file_results = os.path.join(results_dir,f'{FLAGS.modality}-{FLAGS.model}-{score}@{top_cand}.csv')
  df.to_csv(file_results)
  

  reloc_dir = os.path.join('predictions',FLAGS.session,'reloc','fig')
  if not os.path.isdir(reloc_dir):
    os.makedirs(reloc_dir)

  gif_name = os.path.join(reloc_dir, f'{FLAGS.modality}-{FLAGS.model}-{score}@{top_cand}.gif')

  eval.plot(sim_thrs = sim_thres, top = top_cand, record_gif=True, name = gif_name)


  
  