#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from .heads.netvlad import NetVLADLoupe
from .utils import *
from .backbone import resnet, pointnet
from .multihead import MultiHead



class ORCHNet(nn.Module):
  def __init__(self,backbone_name:str,in_channels:int,feat_dim:int,out_dim:int, **argv):
    super(ORCHNet,self).__init__()
    self.backbone_name = backbone_name
    #self.classifier = classifier
    modality = argv.pop('modality')
    if self.backbone_name == 'resnet50':
      return_layers = {'layer4': 'out'}
      #assert 'param' in argv, 'Resnet arguments not available'
      pretrained = argv.pop('pretrained_backbone')
      max_points = argv.pop('max_points')
      backbone = resnet.__dict__[backbone_name](pretrained,**argv)
      self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    else:
      max_points = argv.pop('max_points')
      self.backbone = pointnet.PointNet_features(in_dim=in_channels, dim_k=feat_dim,**argv)


    self.head = MultiHead(outdim=out_dim)
   
  def forward(self,x):
    b = x.shape[0]
    s = x.shape[1]

    y = self.backbone(x)
    y = y['out']
    
    if len(y.shape)>3: # CNN-based output
      b,c,w,h = y.shape
      y = y.reshape(b,c,-1)
      
    if len(y.shape)<4: # Pointnet returns [batch x feature x samples]
      y = y.unsqueeze(dim=-1)
    z = self.head(y)

    return z
  
  def get_backbone_params(self):
    return self.backbone.parameters()

  def get_classifier_params(self):
    return self.head.parameters()
  
  def __str__(self):
    return "ORCHNet_" + self.backbone_name












