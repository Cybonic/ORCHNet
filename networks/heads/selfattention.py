#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F



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


class attention_net(nn.Module):
    def __init__(self,in_dim, n_layers=1,norm_layer=True ):
      super(attention_net,self).__init__()

      depth = n_layers
      in_dim = in_dim
      self.norm = norm_layer

      self.attantion = nn.ModuleList([attention_layer(in_dim) for i in range(depth)])

      if self.norm == True:
        self.norm_layer = nn.LayerNorm([64,16])
    
    def forward(self, x):
      for i, layer in enumerate(self.attantion): 
        x,att = layer(x)
        if self.norm == True:
          x = self.norm_layer(x)
  
      return(x)