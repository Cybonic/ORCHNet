#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..utils import *

'''
Code taken from  https://github.com/slothfulxtx/TransLoc3D 
'''


class MAC(nn.Module):
    def __init__(self,outdim=256, **rgv):
        super().__init__()
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        x = x.view(x.shape[0],x.shape[1],-1)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        return self.fc(x)


class SPoC(nn.Module):
    def __init__(self, outdim=256,**argv):
        super().__init__()
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        x = x.view(x.shape[0],x.shape[1],-1)
        return self.fc(torch.mean(x, dim=-1, keepdim=False)) # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, outdim=256, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        #self.p = p
        self.eps = eps
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # This implicitly applies ReLU on x (clamps negative values)
        x = x.clamp(min=self.eps).pow(self.p)
        
        x = x.view(x.shape[0],x.shape[1],-1)
        x = F.avg_pool1d(x, x.size(-1))
       
        x = x.view(x.shape[0],x.shape[1])
        
        x = torch.pow(x,1./self.p)
        x = self.fc(x)
        return x # Return (batch_size, n_features) tensor

# MultiHead Aggregation
class MultiHead(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MultiHead,self).__init__()
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










