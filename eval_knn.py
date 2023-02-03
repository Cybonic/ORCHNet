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
os.environ['NUMEXPR_NUM_THREADS'] = '8'

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
from utils.retrieval import retrieval_knn
from utils.metric import retrieve_eval

def load_dataset(inputs,session,max_points=50000,debug=False):

    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'

        
    session['val_loader']['data']['modality'] = inputs.modality
    session['val_loader']['data']['sequence'] = inputs.sequence
    session['val_loader']['batch_size'] = inputs.batch_size

    loader = ORCHARDS(root    = session[root_dir],
                        train_loader  = session['train_loader'],
                        val_loader    = session['val_loader'],
                        mode          = inputs.memory,
                        sensor        = sensor_cfg,
                        debug         = debug,
                        max_points = 30000)

    return(loader,[])


def knn_eval(descriptors,anchors,database,true_loop,list_top_cand,eval_metric):


    nra=500 # None Retrieval Area
    max_top = np.max(list_top_cand)

    descriptor_idx = list(descriptors.keys())
    for anchor in anchors:
        database_idx = database[:anchor-nra] # 

        query_dptrs =  np.array([descriptors[i] for i in anchor if i in descriptor_idx ])
        map_dptrs = np.array([descriptors[i] for i in database_idx if i in descriptor_idx ])
        # Retrieve loops 
        retrieved_loops ,scores = retrieval_knn(query_dptrs, map_dptrs, top_cand = max_top, metric = eval_metric)

    # Evaluate retrieval
    overall_scores = {}
    for top in range(max_top):
        scores = retrieve_eval(retrieved_loops,true_loop, top = top)
        overall_scores[top]=scores
    # Post on tensorboard
    return overall_scores,descriptors




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
      '--experiment', '-e',
      type=str,
      required=False,
      default='SamplingOrchardsvSameDataset/P1-N20-NO-TNET_F64/MP-30000/LazyQuadrupletLoss L2/autumn',
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
      default='checkpoints/FITTING/LazyQuadrupletLoss_L2/autumn/VLAD_pointnet/best_model.pth',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--memory',
      type=str,
      required=False,
      default='RAM',
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
      default='orchard-uk',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--sequence',
      type=str,
      required=False,
      default='autumn',
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
      default = 500,
      help='sampling points.'
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
  #print("Dataset  -> Validation: ", SESSION['val_loader']['data']['dataset'])
  #print("Sequence -> Validation: ", SESSION['val_loader']['data']['sequence'])
  print("Memory: ", FLAGS.memory)
  print("Model:  ", FLAGS.model)
  print("Debug:  ", FLAGS.debug)
  print("Resume: ", FLAGS.resume)
  print(f'Device: {FLAGS.device}')
  print(f'batch size: {FLAGS.batch_size}')
  print("----------\n")

 

  dataloader,run_name = load_dataset(FLAGS,SESSION)
                            
  ###################################################################### 
  SESSION['train_loader']['data']['max_points'] = FLAGS.max_points
  SESSION['val_loader']['data']['max_points'] = FLAGS.max_points
  modality = FLAGS.modality + '_param'

  SESSION[modality]['max_samples'] = FLAGS.max_points # For VLAD one as to define the number of samples
  model_ = model.ModelWrapper(**SESSION['model'],loss= [], **SESSION[modality])
  
  run_name['model'] = FLAGS.model
  run_name['experiment'] = FLAGS.experiment
  
  SESSION['retrieval']['top_cand'] = list(range(1,25,1))
  trainer = Trainer(
          model  = model_,
          resume = FLAGS.resume,
          config = SESSION,
          loader = dataloader,
          iter_per_epoch = 10, # Verify this!!!
          device = FLAGS.device,
          run_name = run_name
          )
  

  results,descriptors = trainer.Eval()

  dataset =FLAGS.session
  
  columns = ['top','recall']
  values = [v['recall'] for v in list(results.values())]

  rows = [[t,v] for t,v in zip(list(results.keys()),values)]
  import pandas as pd


  df = pd.DataFrame(rows,columns = columns)
  top = rows[0][0]
  score = round(rows[0][1],2)
  
  results_dir = os.path.join('predictions',dataset,'place','results')
  if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

  file_results = os.path.join(results_dir,f'{FLAGS.modality}-{FLAGS.model}_{score}@{top}.csv')

  df.to_csv(file_results)

  descriptors_dir = os.path.join('predictions',f'{dataset}','place','descriptors')
  if not os.path.isdir(descriptors_dir):
    os.makedirs(descriptors_dir)

  file_name = os.path.join(descriptors_dir,f'{FLAGS.modality}-{FLAGS.model}_{score}@{top}.npy')
  torch.save(descriptors,file_name)

 

  
  