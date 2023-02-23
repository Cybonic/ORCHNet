#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
from ..utils import *
from .pooling import *


# MultiHead Aggregation
class MuHA(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MuHA,self).__init__()
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    
    self.fusion= nn.Parameter(torch.zeros(1,3))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    print(self.fusion.data)


  def forward(self,x):
    
    spoc =  self.spoc(x)
    gem  =  self.gem(x)
    mac  =  self.mac(x)
    z    =  torch.stack([spoc,gem,mac],dim=1)
    fu = torch.matmul(self.fusion,z)

    return fu










