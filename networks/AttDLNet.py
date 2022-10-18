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


class attention_layer(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(attention_layer,self).__init__()
        self.chanel_in = in_dim

        out_dim = in_dim//8 
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
      
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x

        return out,attention

# Attention 

class Attention(nn.Module):
    def __init__(self,in_dim, n_layers=1,norm_layer=False ):
      super(Attention,self).__init__()

      in_dim = in_dim
      self.norm = norm_layer

      self.attantion = nn.ModuleList([attention_layer(in_dim) for i in range(n_layers)])

      if self.norm == True:
        self.norm_layer = nn.LayerNorm([2048, 4, 64])
    
    def forward(self, x):
      for i, layer in enumerate(self.attantion): 
        x,att = layer(x)
        if self.norm == True:
          x = self.norm_layer(x)
      return(x)


# Attention NVLAD Head

class AttVLADHead(nn.Module):
  def __init__(self,in_dim=2048,out_dim=256,max_samples= 256,cluster_size=20,**argv):
    super(AttVLADHead,self).__init__()
    self.model = nn.Sequential(
                  Attention(in_dim=in_dim,norm_layer=False),
                  NetVLADLoupe(feature_size=in_dim, 
                                max_samples=max_samples, 
                                cluster_size=cluster_size,
                                output_dim=out_dim, 
                                gating=True, 
                                add_batch_norm=True,
                                is_training=True),
                  nn.Linear(out_dim,out_dim)
                  )
  def forward(self,x):
    return self.model(x)


class VLADHead(nn.Module):
  def __init__(self,in_dim=2048,out_dim=256,max_samples=128,cluster_size=20,**argv):
      super(VLADHead,self).__init__()
      self.model = nn.Sequential(
                              NetVLADLoupe(feature_size=in_dim, 
                                max_samples=max_samples, 
                                cluster_size=cluster_size,
                                output_dim=out_dim, 
                                gating=True, 
                                add_batch_norm=True,
                                is_training=True),
                              nn.Linear(out_dim,out_dim)
                        )
  def forward(self,x):
    return self.model(x)



class AttDLNet(nn.Module):
  def __init__(self,backbone,classifier):
    super(AttDLNet,self).__init__()
    self.backbone = backbone
    self.classifier = classifier
    
   
  def forward(self,x):
    y = self.backbone(x)
    #y['out'] = torch.flatten(y['out'],start_dim=1)
    y = y['out']
    if len(y.shape)<4: # Pointnet returns [batch x feature x samples]
      y = y.unsqueeze(dim=-1)
      #y = y.transpose(1,3)
    z = self.classifier(y)
    return z
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()











