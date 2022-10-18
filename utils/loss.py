
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
    value = torch.sqrt(torch.sum(squared_diff,dim=dim)+eps)
    return torch.max(value,torch.tensor(eps))

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


def get_distance_function(name):
    if name == 'L2':
        loss = L2_loss 
    elif name == 'cosine':
        loss = cosine_loss
    elif name == 'kl_divergence':
        loss = logit_kl_divergence_loss
    else:
        raise NameError
    return(loss)

# Loss
class LazyTripletLoss():
  def __init__(self, metric= 'L2', margin=0.2 , eps=1e-6,**argv):

    self.margin = margin
    self.metric = metric
    self.eps = torch.tensor(eps)
    
    # Loss types
    self.loss = get_distance_function(metric)

  def __str__(self):
    return type(self).__name__ + ' ' + self.metric
 
  def __call__(self,descriptor = {},**args):
    
    #a_pose,p_pose,n_pose = pose[0],pose[1],pose[2]
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']

    assert a.shape[0] == 1
    assert p.shape[0] == 1

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)
    
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2).squeeze()
    #ap = torch.__dict__[self.reduction](pos_loss_array)
    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    
    an = torch.min(neg_loss_array) # get the lowest negative distance (aka hard)
    s = ap - an + self.margin
    value = torch.max(self.eps,s)
    loss = value.clamp(min=0.0)

    return(loss,{'p':ap,'n':an})


class LazyQuadrupletLoss():
  def __init__(self, metric= 'L2', margin1 = 0.5 ,margin2 = 0.5 , eps=1e-6, **argv):
    
    #assert isinstance(margin,list) 
    #assert len(margin) == 2,'margin has to have 2 elements'

    self.margin1 =  margin1 
    self.margin2 = margin2
    self.metric = metric
    self.eps = torch.tensor(eps)
    # Loss types
    self.loss = get_distance_function(metric)
  
  def __str__(self):
    return type(self).__name__ + ' ' + self.metric


  def __call__(self,descriptor = {},**args):
    
    #a_pose,p_pose,n_pose = pose[0],pose[1],pose[2]
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']
    assert a.shape[0] == 1
    assert p.shape[0] == 1
    assert n.shape[0] >= 2,'negative samples must be at least 2' 

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)

    # Anchor - positive
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2)
    # Anchor - negative
    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    # Hard negative
    n_hard_idx = [torch.argmin(neg_loss_array).cpu().numpy().tolist()] # get the lowest negative distance (aka hard)
    an = neg_loss_array[n_hard_idx]
    # Random negative 
    n_negs    = n_torch.shape[0]
    idx_arr   = np.arange(n_negs)
    elig_neg  = np.setxor1d(idx_arr,n_hard_idx) # Remove from the array the hard negative index
    n_rand_idx  = torch.randint(0,elig_neg.shape[0],(1,)).numpy().tolist()
    dn_hard   = n_torch[n_hard_idx] # Hard negative descriptor
    dn_rand   = n_torch[n_rand_idx] # random negative descriptor
    an_prime= self.loss(dn_hard,dn_rand,dim=2) # distance between hard and random 
    # Compute first tem
    s1 = ap.squeeze() - an.squeeze() + self.margin1
    first_term = torch.max(self.eps,s1).clamp(min=0.0)
    # Compute second term
    s2 = ap.squeeze() - an_prime.squeeze() + self.margin2
    second_term = torch.max(self.eps,s2).clamp(min=0.0)
    # compute loss
    loss = first_term + second_term

    return(loss,{'p':ap,'n':an,'n_p':an_prime})


class LazyTriplet_plus():
    def __init__(self, metric= 'L2', margin=0.2, alpha=0.5, eps=1e-6, add_loss ='kl',**argv ):
        self.add_loss = add_loss
       
        #self.device= metric
        self.margin = margin
        self.metric = metric
        self.eps    = torch.tensor(eps)
        self.alpha  = alpha

        self.loss = L2_loss 
    
    def __str__(self):
        return type(self).__name__ + ' ' + self.metric

    def __call__(self, descriptor = {}, poses = {}):
        
        a_pose,p_pose,n_pose = poses['a'],poses['p'],poses['n']
        a,p,n = descriptor['a'],descriptor['p'],descriptor['n']

        if len(a.shape) < len(p.shape): 
            a = a.unsqueeze(dim=0)
        if len(p.shape) < len(a.shape): 
            p = p.unsqueeze(dim=0)
        if len(n.shape) < len(a.shape): 
            n = n.unsqueeze(dim=0)
      
        # Create query with the same size of the positive
        a_torch,p_torch = totensorformat(a,p)
        dap = self.loss(a_torch, p_torch,dim=2)
        #dap = torch.__dict__[self.reduction](pos_loss_array)

        a_torch,n_torch  = totensorformat(a,n)
        dan = self.loss(a_torch,n_torch,dim=2)
        #dan = torch.__dict__[self.reduction](neg_loss_array)
        an_idx = torch.argmin(dan) # get the lowest n egative distance (aka hard)
        dan = dan[an_idx]

        n_pose = n_pose.transpose(1,0)
        pa_torch,pn_torch  = totensorformat(a_pose,n_pose[an_idx])
        pan = self.loss(pa_torch,pn_torch,dim=2)
        #pan = torch.__dict__[self.reduction](neg_dis)

        pa_torch,pp_torch  = totensorformat(a_pose,p_pose)
        pap = self.loss(pa_torch,pp_torch,dim=2)
        #pap = torch.__dict__[self.reduction](pos_dis)

        #an = torch.__dict__[self.reduction](neg_loss_array)
        delta_an = torch.abs(pan - dan)
        delta_ap = torch.abs(pap - dap)
        kl_ap = logit_kl_divergence_loss(a_torch, p_torch)

        l =  (self.alpha*(delta_an) + (1-self.alpha)*(delta_ap))

        if self.add_loss == 'kl': 
            l += kl_ap

        value = torch.max(self.eps,l)
        loss = value.clamp(min=0.0)

        return(loss,{'p':delta_ap,'n':delta_an})
