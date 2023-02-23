#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .heads.netvlad import NetVLADLoupe
from .utils import *


class ORCHNet(nn.Module):
  def __init__(self,backbone,classifier):
    super(AttDLNet,self).__init__()
    self.backbone = backbone
    self.classifier = classifier
    
   
  def forward(self,x):
    b = x.shape[0]
    s = x.shape[1]

    y = self.backbone(x)
    #y['out'] = torch.flatten(y['out'],start_dim=1)
    y = y['out']
    
    if len(y.shape)>3: # CNN-based output
      b,c,w,h = y.shape
      y = y.reshape(b,c,-1)
      
    if len(y.shape)<4: # Pointnet returns [batch x feature x samples]
      y = y.unsqueeze(dim=-1)
    
    z = self.classifier(y)

    return z
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()












