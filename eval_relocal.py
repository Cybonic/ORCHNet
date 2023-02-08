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
from tqdm import tqdm
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
from mst_trainer import Trainer
from networks import model
from utils.utils import generate_descriptors
from utils.relocalization import relocal_metric,comp_gt_table,sim_relocalize
from utils.viz import myplot
from utils.retrieval import retrieval_knn
#from dataloader.utils import load_dataset


def load_dataset(dataset,session,memory,max_points=None,debug=False):

    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'


    loader = ORCHARDS(root    = session[root_dir],
                        train_loader  = session['train_loader'],
                        test_loader    = session['val_loader'],
                        mode          = memory,
                        split_mode    = 'same', 
                        #subsample     = 0.5
                        )
    
    
    return(loader)


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

def _get_top_cand(pred_idx,pred_scores,pos_thrs=0.5,top=1):
        top_pred_cand_idx = pred_idx[:,:top]
        top_pred_sores = pred_scores[:,:top]
        top_cand_hat = []

        for c,s in zip(top_pred_cand_idx,top_pred_sores):
            c_hat = c[(s<pos_thrs).any() and (s>0).any()]
            top_cand_hat.append(c_hat[0] if len(c_hat)>0 else [])
        return(top_cand_hat)


class Relocalization():
    def __init__(self,model,loader, top_cand, windows=500, metric = 'L2', device = 'cpu'):

        self.loader   = loader
        self.model    = model.to(device)
        self.pose   = loader.dataset.get_pose()
        self.anchor = loader.dataset.get_anchor_idx()
        self.database = loader.dataset.get_idx_universe()
        self.top_cand = top_cand
        self.gt_loop = loader.dataset.get_GT_Map()
        self.device = device
        self.true_loop = np.array([np.where(self.gt_loop[i]==1)[0] for i in range(self.gt_loop.shape[0])])
    
        self.FLAG_USE_DESCRIPTORS = False
        self.metric = metric
        self.windows = windows

    def load_descriptors(self,path):
        assert os.path.isfile(path)
        self.loaded_descriptors = torch.load(self.path_to_descriptors)

    def save_loop_pred_to_csv(self,file):


        df_pred = pd.DataFrame(self.pred_loops)
        df_score = pd.DataFrame(self.pred_scores)
    
        
        results_dir = os.path.join('predictions','reloc','predictions')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        file_results_ped = os.path.join(results_dir,file+'_loops.csv')
        file_results_score = os.path.join(results_dir,file+'_scores.csv')

        df_pred.to_csv(file_results_ped)
        df_score.to_csv(file_results_score)


    def run(self,sim_thres=0.5,burn_in=600):

        # self.gt_table = comp_gt_table(self.pose,self.anchor,range_thres)

        self.model.eval()
        top_max = max(self.top_cand)

        if self.FLAG_USE_DESCRIPTORS:
            descriptors = self.loaded_descriptors
        else:
            descriptors = generate_descriptors(self.model,self.loader,self.device)
            
        # None Retrieval Area
        pred_loops = []
        target_loops = self.true_loop
        descriptor_idx = list(descriptors.keys())
        
        num_samples = self.database.shape[0]
        pred_scores = []
        for anchor in tqdm(range(burn_in,num_samples),'Relocalization'):
    
            database_idx = self.database[:anchor-self.windows] # 
            query_dptrs = np.array([descriptors[i] for i in [anchor] if i in descriptor_idx])
            map_dptrs   = np.array([descriptors[i] for i in database_idx if i in descriptor_idx])

            # Retrieve loops 
            retrieved_loops ,scores = retrieval_knn(query_dptrs, map_dptrs, top_cand = top_max, metric = self.metric)
            pred_loops.append(retrieved_loops[0])
            pred_scores.append(scores[0])

        self.pred_loops = np.array(pred_loops)
        self.pred_scores = np.array(pred_scores)
        overall_scores = {}

        for top in self.top_cand:
            top_cand_hat = _get_top_cand(self.pred_loops,self.pred_scores,pos_thrs=sim_thres,top=top)
            scores = relocal_metric(top_cand_hat,target_loops)
            overall_scores[top] = scores
            print(scores)
        
        return(overall_scores)

    
    def plot(self, sim_thrs = 0.5, record_gif=False, top=25, name= 'relocalization.gif'):

        top_cand_hat = _get_top_cand(self.pred_loops,self.pred_scores,pos_thrs=sim_thrs,top=top)
        tp_idx,fp_idx = comp_eval_idx(top_cand_hat,self.true_loop)

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
      default='SPoC_pointnet',
      help='Directory to get the trained model.'
  )
  
  parser.add_argument(
      '--experiment', '-e',
      type=str,
      required=False,
      default=None,
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
      default='checkpoints/RelocTrainF128P10k-LazyQuadrupletLoss_L2-autumn-SPoC_pointnet-recall@20-45.pth',
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
      '--plot',
      type=int,
      required=False,
      default=1,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--session',
      type=str,
      required=False,
      default='orchard-uk',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--sequence',
      type=str,
      required=False,
      default='summer',
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
      '--batch_size',
      type=int,
      required=False,
      default=10,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--max_points',
      type=int,
      required=False,
      default = 10000,
      help='sampling points.'
  )
  parser.add_argument(
      '--modality',
      type=str,
      required=False,
      default='pcl', # [pcl,bev, projection]
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
  SESSION['train_loader']['data']['modality'] = FLAGS.modality
  SESSION['val_loader']['data']['modality'] = FLAGS.modality
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
  print(f'Experiment: {FLAGS.experiment}')
  print("----------\n")

  descriptor_root = f'predictions/{FLAGS.session}/descriptors'
  if not os.path.isdir(descriptor_root):
    os.makedirs(descriptor_root)

  descriptor_path = None
  if FLAGS.experiment != None:
    descriptor_path = os.path.join(descriptor_root,FLAGS.experiment+'.npy')
    assert os.path.isfile(descriptor_path),'Descriptor File does not exist!'

  else:
    experiment = '-'.join(FLAGS.resume.split(os.sep)[1:-1])
    descriptor_path = os.path.join(descriptor_root,experiment+'.npy')

   ###################################################################### 
   # open arch config file
  cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
  print("Opening data config file: %s" % cfg_file)
  sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))

  loader = load_dataset(FLAGS,SESSION,FLAGS.memory,debug = FLAGS.debug)


  SESSION['train_loader']['data']['max_points'] = FLAGS.max_points
  SESSION['val_loader']['data']['max_points'] = FLAGS.max_points
  modality = FLAGS.modality + '_param'

  SESSION[modality]['max_samples'] = FLAGS.max_points # For VLAD one as to define the number of samples
 
  model_ = model.ModelWrapper(**SESSION['model'],loss= [], **SESSION[modality])

  checkpoint = torch.load(FLAGS.resume,map_location=torch.device(FLAGS.device))
  
  try: 
    model_.load_state_dict(checkpoint['state_dict'])
    print("\n**********************************************************")
    print("Just loaded a pre-trained model")
    print("Recall: " + str(checkpoint['monitor_best']['recall']))
    print("**********************************************************\n")
  except:
    print("No Model Loaded!")
    exit()

  
  run_name = {  'dataset': SESSION['train_loader']['data']['dataset'],
                'dataset': SESSION['train_loader']['data']['sequence'],
                'experiment':SESSION['experim_name'], 
                'model':SESSION['model']['type']
                }


  
 
 ###################################################################### 
  eval = Relocalization(
          model  = model_,
          loader = loader.get_val_loader(),
          device = FLAGS.device,
          top_cand= [1,5,25]
          )
  
  sim_thres = 0.5 
  # Compute performance
  results = eval.run(   sim_thres,
                        burn_in  = 600
                        )
  

  columns = ['top','recall','precision']
  rows = [[t,v['recall'],v['precision']] for t,v in results.items()]
  import pandas as pd 
  
  top_cand = 25
  score =  round(results[top_cand]['recall'],3)

  file = f'{FLAGS.modality}-{FLAGS.model}_{score}'
  eval.save_loop_pred_to_csv(file)
  # Save Results
  top_cand = 25
  score =  round(results[top_cand]['recall'],3)
  
  df = pd.DataFrame(rows,columns = columns)

  # Build file name for all formats
  file_name = f'{experiment}-{score}@{top_cand}'
  # Save relocalization Results
  results_dir = os.path.join('predictions',FLAGS.session,'results')
  if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

  file_results = os.path.join(results_dir,'reloc' + file_name + '.csv')
  df.to_csv(file_results)
  
  # save relocalization Gif
  reloc_dir = os.path.join('predictions',FLAGS.session,'reloc')
  if not os.path.isdir(reloc_dir):
    os.makedirs(reloc_dir)

  gif_name = os.path.join(reloc_dir,file_name+'.gif')

  eval.plot(sim_thrs = sim_thres, top = top_cand, record_gif=False, name = gif_name)


  
  