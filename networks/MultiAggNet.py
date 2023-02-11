#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
#from .heads.netvlad import NetVLADLoupe

#from .utils import TripletLoss
import torch.nn.init as init
from .heads.pooling import *
from .utils import *

class MuANet(nn.Module):
  def __init__(self,backbone,hiddendim=64,outdim=256):
    super(MuANet,self).__init__()
    self.backbone = backbone
    #self.classifier = classifier
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    
    self.fusion= torch.nn.Parameter(torch.zeros(1,3))
    #self.out = nn.LazyLinear((1,256))
    nn.init.normal_(self.fusion.data, mean=0, std=0.1)
    print(self.fusion.data)

    self.mean = nn.Linear(hiddendim,hiddendim)
    self.std = nn.Linear(hiddendim,hiddendim)

    
  def sampler(z_mean,z_varstd):
  
    shape = z_mean.shape
    epsilon = torch.randn(shape=shape)
    return z_mean + torch.exp(0.5 * z_varstd) * epsilon

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
    b,f,p,h = y.shape
    #y = y.flatten(dim=1)
    #z_mean=self.mean(y)
    #z_var = self.std(y)

    #z1 = self.sampler(z_mean,z_var)
    #z2 = self.sampler(z_mean,z_var)
    #z3 = self.sampler(z_mean,z_var)

    
    spoc = F.normalize(self.spoc(y),dim=1)
    gem  =  F.normalize(self.gem(y),dim=1)
    mac  =  F.normalize(self.mac(y),dim=1)

    z    =  torch.stack([spoc,gem,mac],dim=1)
    #zz = self.out(z)
    y = torch.matmul(self.fusion,z)
    mac  =  F.normalize(y,dim=1)
    #y = torch.matmul(self.fusion, z)
    #z = torch.mean(z,dim=0)
    # z = self.classifier(y)
    #z = F.softmax(z,dim=1)
    #s = z.sum(dim=1)
    # output =  z_norm.reshape((b,s,256))
    return y
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return []# self.classifier.parameters()











