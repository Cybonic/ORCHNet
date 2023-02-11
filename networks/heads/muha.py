#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .netvlad import NetVLADLoupe
from ..utils import *
#from .utils import TripletLoss
import torch.nn.init as init
from .pooling import *


# MultiHead Aggregation
class MuHA(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MuHA,self).__init__()
    #self.classifier = classifier
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    
    self.fusion= torch.nn.Parameter(torch.zeros(1,3))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    print(self.fusion.data)


  def forward(self,x):
    
    #spoc =  F.normalize(self.spoc(x),dim=1)
    #gem  =  F.normalize(self.gem(x),dim=1)
    #mac  =  F.normalize(self.mac(x),dim=1)
    
    spoc =  self.spoc(x)
    gem  =  self.gem(x)
    mac  =  self.mac(x)
    
    z    =  torch.stack([spoc,gem,mac],dim=1)

    fu = torch.matmul(self.fusion,z)
    #y  =  F.normalize(fu,dim=1)

    return fu










