import torch
import torch.nn as nn
from networks import modeling 
import torchmetrics
import os
import math


class ModelWrapper(nn.Module):
    def __init__(self,  type,
                        loss        = None,
                        output_dim  = 256, 
                        minibatch_size = 3, 
                        pretrained_backbone= False,  
                        device = 'cuda',
                        **args,
                        ):
                        
        super(ModelWrapper,self).__init__()
        assert minibatch_size>= 3, 'Minibatch size too small'
        
        self.loss = loss
        self.minibatch_size=minibatch_size
        self.device = device

        self.model = modeling.__dict__[type](output_dim   = output_dim, 
                                            output_stride = 4, 
                                            pretrained_backbone = pretrained_backbone,
                                            **args )


    def forward(self,pcl,):
        # Mini Batch training due to memory constrains
        if self.training == False:
            return(self.model(pcl))

        anchor,positive,negative = pcl[0]['anchor'],pcl[0]['positive'][0],pcl[0]['negative'][0]
        pose_anchor,pose_positive,pose_negative = pcl[1]['anchor'],pcl[1]['positive'],pcl[1]['negative']
        num_anchor,num_pos,num_neg = anchor.shape[0],positive.shape[0],negative.shape[0]
        
        pose = {'a':pose_anchor,'p':pose_positive,'n':pose_negative}
        
        batch_loss = 0
        #self.minibatch_size = 20
        mini_batch_total_iteration = math.ceil(num_neg/self.minibatch_size)
        for i in range(0,mini_batch_total_iteration): # This works because neg < pos
            
            j = i*self.minibatch_size
            k = j + self.minibatch_size
            if k > num_neg:
                k = j + (num_neg - j)

            neg = negative[j:k]
            pclt = torch.cat((anchor,positive,neg))
            
            if pclt.shape[0]==1: # drop last
                continue

            pred = self.model(pclt)
            
            a_idx = num_anchor
            p_idx = num_pos+num_anchor
            n_idx = num_pos+num_anchor + num_neg
            
            dq,dp,dn = pred[0:a_idx],pred[a_idx:p_idx],pred[p_idx:n_idx]
            descriptor = {'a':dq,'p':dp,'n':dn}

            loss_value,info = self.loss(descriptor = descriptor, poses = pose)
            # devide by the number of batch iteration; as direct implication in the grads
            loss_value /= mini_batch_total_iteration 
            loss_value.backward() # Backpropagate gradients and free graph
            batch_loss += loss_value.detach().cpu().item()

        return({'l':loss_value,**info})
    
    def get_backbone_params(self):
        return self.model.get_backbone_params()

    def get_classifier_params(self):
        return self.model.get_classifier_params()
    
    def resume(self,path):
        assert os.path.isfile(path),'Something is work with the path: '+ path
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Loader pretrained model: " + path)

    
