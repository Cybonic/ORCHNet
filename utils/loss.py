
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================================
#  KL divergence
# ==========================================================================
def pmf(input_tensor,tau = 1, eps=1e-6):
    log_probs = F.log_softmax(input_tensor/tau,dim=2)
    exp = torch.exp(log_probs).clone()
    exp[exp==0]=eps
    return(exp)

def logit_kl_divergence_loss(x, y, eps=1e-6, **argv):
    # Map to probabilistic mass function
    px = pmf(x)
    py = pmf(y)
    kl = px * torch.log2(px / py)
    return torch.max(torch.nansum(kl,dim=[2,1]),torch.tensor(eps))

def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = torch.nansum(kl,dim=[2,1])
    return out

def kl_divergence_loss(anchor,positive,negative, eps=1e-6,**arg):
    # Compute Anchor->negative KLD
    a_torch,p_torch = totensorformat(anchor,positive)
    ap = logit_kl_divergence(a_torch,p_torch).mean()
    # Compute Anchor->negative KLD
    a_torch,n_torch = totensorformat(anchor,negative)
    an = logit_kl_divergence(a_torch,n_torch).mean()
    #similar_percentage =(torch.abs(xy - xz) / ((xy + xz)/2))
    
    return(ap,an)

# ==========================================================================
#  cosine
# ==========================================================================
def cosine_loss(x,y,eps=1e-6,dim=0):
    return torch.max(1-torch.abs(cosine(x,y,dim)),torch.tensor(eps))

def cosine(a,b,dim=0):
    num = torch.tensordot(b,a,dims=([1,2],[1,2])).squeeze()
    den = (torch.norm(a,dim=2)*torch.norm(b,dim=2)).squeeze()
    return torch.div(num,den)

# ==========================================================================
#  Euclidean Distance
# ==========================================================================

def L2_np(a,b, dim=0):
    return np.sqrt(np.sum((a - b)**2,axis=dim))

def L2_loss(a,b, dim=0, eps=1e-6):
    squared_diff = torch.pow((a - b),2)
    return torch.max(torch.sqrt(torch.sum(squared_diff,dim=dim)).squeeze(),torch.tensor(eps))

def totensorformat(x,y):
    if not torch.is_tensor(x):
        x = torch.tensor(x,dtype=torch.float32)

    if len(x.shape)<3:
        bs = x.shape[0]
        x = x.view((bs,1,-1))

    if not torch.is_tensor(y):
        y = torch.tensor(y,dtype=torch.float32)

    if len(y.shape)<3:
        bs = y.shape[0]
        y = y.view(bs,1,-1)
    
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    return(x,y)


# Loss
class TripletLoss():
  def __init__(self, metric= 'L2', margin=0.2 , reduction='sum', use_min=False, eps=1e-6,mode='distro'):
    
    self.use_min = use_min
    assert reduction in ['sum','mean']
    self.reduction = reduction
    #self.device= metric
    self.margin = margin
    self.metric = metric
    self.eps = torch.tensor(eps)
    self.mode = mode
    
    # Loss types
    if metric == 'L2':
        self.loss = L2_loss 
    elif metric == 'cosine':
        self.loss = cosine_loss
    elif metric == 'kl_divergence':
        self.loss = logit_kl_divergence_loss
    
  def __call__(self,a,p,n):
  
    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)
    
    # Create query with the same size of the positive
    a_torch,p_torch = totensorformat(a,p)
    pos_loss_array = self.loss(a_torch, p_torch,dim=2)
    ap = torch.__dict__[self.reduction](pos_loss_array)

    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    an = torch.__dict__[self.reduction](neg_loss_array)

    s = ap - an + self.margin

    value = torch.max(self.eps,s)
    loss = value.clamp(min=0.0)

    return({'l':loss,'p':ap,'n':an})

        # result = torch.ones(dneg.shape[0]).to(self.device)*dp + self.margin - dneg


class DoubleTriplet():
    def __init__(self, metric= 'L2', margin=0.2, alpha=0.5, reduction='sum', use_min=False, eps=1e-6,mode='distro'):
        self.use_min = use_min
        assert reduction in ['sum','mean']
        self.reduction = reduction
        #self.device= metric
        self.margin = margin
        self.metric = metric
        self.eps = torch.tensor(eps)
        self.mode = mode
        self.alpha=alpha

        self.loss = L2_loss 

    
    def __call__(self,a,p,n,a_pose,p_pose,n_pose):
  
        if len(a.shape) < len(p.shape): 
            a = a.unsqueeze(dim=0)
        if len(p.shape) < len(a.shape): 
            p = p.unsqueeze(dim=0)
        if len(n.shape) < len(a.shape): 
            n = n.unsqueeze(dim=0)
      
        
        # Create query with the same size of the positive
        a_torch,p_torch = totensorformat(a,p)
        pos_loss_array = self.loss(a_torch, p_torch,dim=2)
        ap = torch.__dict__[self.reduction](pos_loss_array)

        a_torch,n_torch  = totensorformat(a,n)
        neg_loss_array = self.loss(a_torch,n_torch,dim=2)
        an = torch.__dict__[self.reduction](neg_loss_array)

        a_torch,n_torch  = totensorformat(a_pose,n_pose)
        neg_dis = self.loss(a_torch,n_torch)
        neg_dis = torch.__dict__[self.reduction](neg_dis)

        a_torch,p_torch  = totensorformat(a_pose,p_pose)
        pos_dis = self.loss(a_torch,p_torch)
        pos_dis = torch.__dict__[self.reduction](pos_dis)

        #an = torch.__dict__[self.reduction](neg_loss_array)

        l =  (self.alpha*(neg_dis - an).abs() + (1-self.alpha)*(pos_dis - ap).abs())
        
        value = torch.max(self.eps,l)
        loss = value.clamp(min=0.0)

        return({'l':loss,'p':ap,'n':an})
