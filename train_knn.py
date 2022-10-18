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
from dataloader.KITTI import KITTI
from trainer import Trainer
from networks import model
from utils import loss as losses



def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def load_dataset(dataset,session,memory,max_points=50000,debug=False):

    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'


    if dataset == 'kitti':
        
        if debug:
            session['train_loader']['data']['sequence'] = ['00']
            session['val_loader']['data']['sequence'] = ['00']
            print("[Main] Debug mode ON: training and Val on Sequence 00 \n")

        loader = KITTI( root = session[root_dir],
                        train_loader  = session['train_loader'],
                        val_loader    = session['val_loader'],
                        mode          = memory,
                        sensor        = sensor_cfg,
                        debug         = debug,
                        max_points = 50000)
    else:

        loader = ORCHARDS(root    = session[root_dir],
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = memory,
                            sensor        = sensor_cfg,
                            debug         = debug,
                            max_points = 30000)
    
    return(loader)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default=  'VLAD_pointnet',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--experiment', '-e',
      type=str,
      required=False,
      default='similarityLoss/bev/alpha0.5/hard',
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
      '--resume', '-r',
      type=str,
      required=False,
      default='None',
              #'/home/tiago/Dropbox/research-projects/orchards-uk/src/AttDLNet/checkpoints/range-rerecord_sparce-AttVLAD_resnet50-0.87.pth',
              #'/home/tiago/Dropbox/research-projects/orchards-uk/src/AttDLNet/checkpoints/bev-rerecord_sparce-AttVLAD_resnet50-0.54.pth',
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
      '--epoch',
      type=int,
      required=False,
      default=100,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--modality',
      type=str,
      required=False,
      default='pcl', # [pcl,bev, projection]
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--dataset',
      type=str,
      required=False,
      default='kitti',
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
      default=20,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--mini_batch_size',
      type=int,
      required=False,
      default=50, #  Max size (based on the negatives)
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--debug',
      type=bool,
      required=False,
      default=True,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--loss',
      type=str,
      required=False,
      default = 'LazyTriplet_plus',
      choices = ['LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss'],
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  
  # Configuration conditions that have to be met
  assert not (FLAGS.modality=='bev' and FLAGS.model.endswith('pointnet')), 'BEV modality does not work with pointnet'
  assert not (FLAGS.modality=='pcl' and  'resnet' in FLAGS.model), 'PCL modality does not work with resnet'
  
  #torch.cuda.empty_cache()
  torch.autograd.set_detect_anomaly(True)
  #force_cudnn_initialization()
  

  # open arch config file
  cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
  print("Opening data config file: %s" % cfg_file)
  sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))

  session_cfg_file = os.path.join('sessions', FLAGS.dataset + '.yaml')
  print("Opening session config file: %s" % session_cfg_file)
  SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
  
  # Update config file with new settings
  SESSION['model']['minibatch_size']  = FLAGS.mini_batch_size
  SESSION['val_loader']['batch_size'] = FLAGS.batch_size
  SESSION['train_loader']['data']['modality'] = FLAGS.modality
  SESSION['val_loader']['data']['modality'] = FLAGS.modality
  SESSION['model']['type'] =  FLAGS.model
  SESSION['trainer']['epochs'] =  FLAGS.epoch
  SESSION['loss']['type'] = FLAGS.loss

  print("----------")
  print("Root: ", SESSION['root'])
  print("\n======= TRAIN LOADER =======")
  print("Dataset  : ", SESSION['train_loader']['data']['dataset'])
  print("Sequence : ", SESSION['train_loader']['data']['sequence'])
  print("\n======= VAL LOADER =======")
  print("Dataset  : ", SESSION['val_loader']['data']['dataset'])
  print("Sequence : ", SESSION['val_loader']['data']['sequence'])
  print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
  print("\n========== MODEL =========")
  print("Model : ", FLAGS.model)
  print("Resume: ", FLAGS.loss)
  print("Loss: ", FLAGS.resume)
  print("MiniBatch Size: ", str(SESSION['model']['minibatch_size']))
  print("\n==========================")
  print(f'Memory: {FLAGS.memory}')
  print(f'Device: {FLAGS.device}')
  print("Loss: %s" %(SESSION['loss']['type']))
  print("Experiment: %s" %(FLAGS.experiment))
  print("Max epochs: %s" %(FLAGS.epoch))
  print("Modality: %s" %(FLAGS.modality))
  print("----------\n")
  
  # For repeatability
  torch.manual_seed(0)
  np.random.seed(0)
  ###################################################################### 
  # Load Dataset
  orchard_loader = load_dataset(FLAGS.dataset,SESSION,FLAGS.memory,debug = FLAGS.debug)
  # Get Loss parameters
  loss_type  = SESSION['loss']['type']
  loss_param = SESSION['loss']['args']
  # Load the loss function
  loss = losses.__dict__[loss_type](**loss_param)
  # Get model parameters based on the modality
  modality = FLAGS.modality + '_param'
  # Load the model
  model_ = model.ModelWrapper(**SESSION['model'],loss= loss, **SESSION[modality])
  

  run_name = {  'dataset': str(SESSION['train_loader']['data']['sequence']),
                'experiment':os.sep.join([FLAGS.loss,FLAGS.modality]), 
                'model': FLAGS.model
                }

  trainer = Trainer(
          model  = model_,
          resume = FLAGS.resume,
          config = SESSION,
          loader = orchard_loader,
          iter_per_epoch = 10, # Verify this!!!
          device = FLAGS.device,
          run_name = run_name
          )
  
  trainer.Train()

  
  