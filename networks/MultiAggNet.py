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
#from .utils import TripletLoss

from .heads.pooling import *

class MuANet(nn.Module):
  def __init__(self,backbone,outdim=256):
    super(MuANet,self).__init__()
    self.backbone = backbone
    #self.classifier = classifier
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
   
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
    
    #y = y.reshape((b,-1,1024)) # <- You have to change this
      #y = y.transpose(1,3)

    z = [self.spoc(y), self.gem(y), self.mac(y)]

    
    torch.mean(z)
    # z = self.classifier(y)
    #z = F.softmax(z,dim=1)
    #s = z.sum(dim=1)
    # output =  z_norm.reshape((b,s,256))
    return z
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()











