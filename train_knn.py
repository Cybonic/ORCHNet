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
import os
import torch 

from networks.AttDLNet import *

from dataloader.ORCHARDS import ORCHARDS
from trainer import Trainer
from networks import model
from utils import loss as losses



def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def load_dataset(dataset,session,memory,max_points=None,debug=False):

    # To get Windows or ubuntu paths
    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'


    loader = ORCHARDS(root    = session[root_dir],
                        train_loader  = session['train_loader'],
                        test_loader    = session['val_loader'],
                        mode          = memory,
                        sensor        = sensor_cfg,
                        split_mode    = 'train-test', # ['train-test','cross-val']
                        )
    
    return(loader)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='GeM_pointnet',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--experiment', '-e',
      type=str,
      required=False,
      default='FINETUNE_RESNETD512',
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
      '--epoch',
      type=int,
      required=False,
      default=30,
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
      default='orchard-uk', #
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
      '--mini_batch_size',
      type=int,
      required=False,
      default=30, #  Max size (based on the negatives)
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--debug',
      type=bool,
      required=False,
      default=False,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--loss',
      type=str,
      required=False,
      default = 'LazyQuadrupletLoss',
      #choices = ['MetricLazyQuadrupletLoss','LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss'],
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--loss_alpha',
      type=float,
      required=False,
      default = 0.5,
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
  
  # Configuration conditions that have to be met
  assert not (FLAGS.modality=='bev' and FLAGS.model.endswith('pointnet')), 'BEV modality does not work with pointnet'
  assert not (FLAGS.modality=='pcl' and  'resnet' in FLAGS.model), 'PCL modality does not work with resnet'
  
  torch.cuda.empty_cache()
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
  SESSION['loss']['args']['alpha'] = FLAGS.loss_alpha

  SESSION['train_loader']['data']['max_points'] = FLAGS.max_points
  SESSION['val_loader']['data']['max_points'] = FLAGS.max_points

  print("----------")
  print("Root: ", SESSION['root'])
  print("\n======= TRAIN LOADER =======")
  print("Dataset  : ", SESSION['train_loader']['data']['dataset'])
  print("Sequence : ", SESSION['train_loader']['data']['sequence'])
  print("Max Points: " + str(SESSION['train_loader']['data']['max_points']))
  print("\n======= VAL LOADER =======")
  print("Dataset  : ", SESSION['val_loader']['data']['dataset'])
  print("Sequence : ", SESSION['val_loader']['data']['sequence'])
  print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
  print("Max Points: " + str(SESSION['val_loader']['data']['max_points']))
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

  loss = losses.__dict__[loss_type](**loss_param,device = FLAGS.device)
  
  print("*"*30)
  print(f'Loss: {loss}')
  print("*"*30)

  # Get model parameters based on the modality
  modality = FLAGS.modality + '_param'
  # Load the model
  SESSION[modality]['max_samples'] = FLAGS.max_points # For VLAD one as to define the number of samples
  model_ = model.ModelWrapper(**SESSION['model'],loss= loss, **SESSION[modality])
  
  run_name = {  'dataset': str(SESSION['train_loader']['data']['sequence']),
                'experiment':os.path.join(FLAGS.experiment,str(loss)), 
                'model': FLAGS.model
                }

  trainer = Trainer(
          model  = model_,
          resume = FLAGS.resume,
          config = SESSION,
          loader = orchard_loader,
          iter_per_epoch = 10, # Verify this!!!
          device = FLAGS.device,
          run_name = run_name,
          train_epoch_zero = True 
          )
  
  trainer.Train()

  
  