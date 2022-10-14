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


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='VLAD_resnet50',
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
      default='checkpoints/range-rerecord_sparce-VLAD_resnet50-0.81.pth',
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
      default='range',
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
      default='rerecord_sparce',
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
      default=1,
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


  print("----------")
  print("INTERFACE:")
  print("Root: ", SESSION['root'])
  print("Dataset  -> Validation: ", SESSION['val_loader']['data']['dataset'])
  print("Sequence -> Validation: ", SESSION['val_loader']['data']['sequence'])
  print("Memory: ", FLAGS.memory)
  print("Model:  ", FLAGS.model)
  print("Debug:  ", FLAGS.debug)
  print("Resume: ", FLAGS.resume)
  print(f'Device: {FLAGS.device}')
  print(f'batch size: {FLAGS.batch_size}')
  print("----------\n")

  SESSION['val_loader']['data']['modality'] = FLAGS.modality
  SESSION['val_loader']['data']['sequence'] = FLAGS.sequence
  SESSION['val_loader']['batch_size'] = FLAGS.batch_size

  orchard_loader = ORCHARDS(root       = SESSION['root'],
                            val_loader = SESSION['val_loader'],
                            sensor     = sensor_cfg,
                            mode       = FLAGS.memory,
                            debug      = FLAGS.debug)
                            
  ###################################################################### 
  modality = FLAGS.modality + '_param'
  model_ = model.ModelWrapper(FLAGS.model, loss=None , device= FLAGS.device,**SESSION[modality])

  run_name = {  'dataset': SESSION['train_loader']['data']['sequence'],
                'experiment':SESSION['experim_name'], 
                'model':SESSION['model']['type']
                }

  SESSION['retrieval']['top_cand'] = list(range(1,25,1))
  trainer = Trainer(
          model  = model_,
          resume = FLAGS.resume,
          config = SESSION,
          loader = orchard_loader,
          iter_per_epoch = 10, # Verify this!!!
          device = FLAGS.device,
          run_name = run_name
          )
  

  results,descriptors = trainer.Eval()

  dataset = SESSION['train_loader']['data']['dataset']
  
  columns = ['top','recall']
  values = [v['recall'] for v in list(results.values())]

  rows = [[t,v] for t,v in zip(list(results.keys()),values)]
  import pandas as pd


  df = pd.DataFrame(rows,columns = columns)
  top = rows[0][0]
  score = round(rows[0][1],2)
  df.to_csv(f'predictions/{dataset}/results/{FLAGS.sequence}{FLAGS.modality}-{FLAGS.model}_{score}@{top}.csv')

  file_name = os.path.join('predictions',f'{dataset}','descriptors',f'{FLAGS.sequence}-{FLAGS.modality}-{FLAGS.model}_{score}@{top}.npy')
  torch.save(descriptors,file_name)

 

  
  