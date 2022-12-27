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

        out_dim = in_dim//16 
    
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1,dilation=2)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1,dilation=2)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1,dilation=2)
        # = nn.Conv2d(in_channels = out_dim , out_channels = in_dim , kernel_size= 1)
        self.upscale = torch.nn.ConvTranspose2d(1,1,3,stride=16)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=1) #
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,points,features = x.size()
        ##print("TEST")
        ##print(x.shape)
        #m_batchsize,C,width ,height = x.size()
        x1 = x.view(m_batchsize,features,-1,1).clone() # B X CX(N)
        # print(x1.shape)
        proj_query  = self.query_conv(x1).squeeze().clone()#.view(m_batchsize,-1,features)
        ##print(proj_query.shape)
        proj_key =  self.key_conv(x1).squeeze()
        ##print(proj_key.shape)
        b,f,p = proj_key.size()
        
        proj_key = proj_key.view(m_batchsize,-1,f) # B X C x (*W*H)
        ##print(proj_key.shape)
        
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        #attention = self.softmax(energy) # BX (N) X (N)
        #debug = torch.sum(attention,dim=1)
        #print(debug[1,0:5])
        b,w,h = energy.size() 
        energy = energy.view(m_batchsize,1,w,h)
        re_scale = self.upscale(energy, output_size=(m_batchsize,1,features,features)).squeeze()#.view(m_batchsize,-1,features) # B X C X N
        proj_value = self.value_conv(x1).view(m_batchsize,-1,features) # B X C X N
        
        attention = self.softmax(re_scale)
        out = torch.bmm(proj_value,attention) # .permute(0,2,1) 
      
        out = out.view(m_batchsize,points,features)
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
                                is_training=True)
                  #nn.Linear(out_dim,out_dim)
                  )
  def forward(self,x):
    return self.model(x)


class VLADHead(nn.Module):
  def __init__(self,in_dim=2048,out_dim=256,max_samples=128,cluster_size=64,**argv):
      super(VLADHead,self).__init__()
      self.model = nn.Sequential(
                              NetVLADLoupe(feature_size=in_dim, 
                                max_samples=max_samples, 
                                cluster_size=cluster_size,
                                output_dim=out_dim, 
                                gating=True, 
                                add_batch_norm=True,
                                is_training=True)
                             # nn.Linear(out_dim,out_dim)
                        )
  def forward(self,x):
    return self.model(x)



class AttDLNet(nn.Module):
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
    if len(y.shape)<4: # Pointnet returns [batch x feature x samples]
      y = y.unsqueeze(dim=-1)
    
    y = y.reshape((b,-1,1024)) # <- You have to change this
    y_nom= F.normalize(y, p=2.0, dim=1, eps=1e-12, out=None)
      #y = y.transpose(1,3)
    z = self.classifier(y_nom)
    z_norm= F.normalize(z, p=2.0, dim=1, eps=1e-12, out=None)
    # output =  z_norm.reshape((b,s,256))
    return z_norm
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()
  
  def get_backbone_params(self):
        return self.backbone.parameters()

  def get_classifier_params(self):
      return self.classifier.parameters()











