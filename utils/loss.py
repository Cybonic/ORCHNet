
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================================
#  KL divergence
# ==========================================================================
def pmf(input_tensor,tau = 1, eps=1e-8):
    log_probs = F.log_softmax(input_tensor/tau,dim=2)
    exp = torch.exp(log_probs).clone()
    exp[exp==0]=eps
    return(exp)

def logit_kl_divergence_loss(x, y, eps=1e-8, **argv):
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
def cosine_torch_loss(x,y,eps=1e-8,dim=0):
    #return torch.max(1-torch.abs(cosine(x,y,dim)),torch.tensor(eps))
    loss = 1 - F.cosine_similarity(x, y, dim, eps)
    return torch.max(loss,torch.tensor(eps))

def cosine_loss(x,y,eps=1e-8,dim=0):
    return torch.max(1-torch.abs(cosine(x,y,dim)),torch.tensor(eps))
    #value = 1-F.cosine_similarity(x, y, dim, eps)
    #return torch.max(value,torch.tensor(eps))
    
def cosine(a,b,dim=0):
    num = torch.tensordot(b,a,dims=([1,2],[1,2])).squeeze()
    den = (torch.norm(a,dim=2)*torch.norm(b,dim=2)).squeeze()
    return torch.div(num,den)

# ==========================================================================
#  Euclidean Distance
# ==========================================================================

def L2_np(a,b, dim=0):
    return np.sqrt(np.sum((a - b)**2,axis=dim))

def L2_loss(a,b, dim=0, eps=1e-8):
    squared_diff = torch.pow((a - b),2)
    value = torch.sqrt(torch.sum(squared_diff,dim=dim)+eps)
    return torch.max(value,torch.tensor(eps))

class L2Loss():
    def __ini__(self,reduction='mean',dim=0):
        self.reduction = reduction
        self.dim=dim
    
    def _call__(self,input,target):
        value = L2_loss(input,target, dim=0, eps=1e-8)
        return torch.__dict__[self.reduction](value)

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
    if name == 'L2':              loss = L2_loss 
    elif name == 'cosine':        loss = cosine_loss
    elif name == 'cosine_torch':  loss = cosine_torch_loss
    elif name == 'kl_divergence': loss = logit_kl_divergence_loss
    elif name == 'kernel_product': loss = kernel_product
    else:
        raise NameError
    return(loss)

#==================================================================================================
#
#
#==================================================================================================

class LazyTripletLoss():
  def __init__(self, metric= 'L2', margin=0.2 , eps=1e-8,**argv):

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
    assert p.shape[0] == 1, 'positives samples must be 1'

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

#==================================================================================================
#
#
#==================================================================================================

class LazyQuadrupletLoss():
  def __init__(self, metric= 'L2', margin1 = 0.5 ,margin2 = 0.5 , eps=1e-8, **argv):
    
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
    assert p.shape[0] == 1, 'positives samples must be 1'
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
    n_hard_idx = [torch.argmin(neg_loss_array).cpu().numpy().tolist()] # get the negative with smallest distance (aka hard)
    an = neg_loss_array[n_hard_idx]
    
    # Random negative (NR)
    n_negs    = n_torch.shape[0]
    idx_arr   = np.arange(n_negs)
    elig_neg  = np.setxor1d(idx_arr,n_hard_idx) # Remove from the negative array the hard negative index
    n_rand_idx  = torch.randint(0,elig_neg.shape[0],(1,)).numpy().tolist()
    dn_hard   = n_torch[n_hard_idx] # Hard negative descriptor
    dn_rand   = n_torch[n_rand_idx] # random negative descriptor
    #nr2h= self.loss(dn_hard,dn_rand,dim=2) # among the negatives select subset of eligibles, and compute the distance between NR and all negatives  
    
    nn_prime= self.loss(n_torch[elig_neg],dn_rand,dim=2) # among the negatives select subset of eligibles, and compute the distance between NR and all negatives  
    n_random_hard_idx = [torch.argmin(nn_prime).cpu().numpy().tolist()] # get the negative with smallest distance w.r.t NR (aka NRH)
    nr2h = nn_prime[n_random_hard_idx] # get the smallest distance between NR and NRH

    # Compute first term
    s1 = ap.squeeze() - an.squeeze() + self.margin1
    first_term = torch.max(self.eps,s1).clamp(min=0.0)
    # Compute second term
    s2 = ap.squeeze() - nr2h.squeeze() + self.margin2
    second_term = torch.max(self.eps,s2).clamp(min=0.0)
    # compute loss
    loss = first_term + second_term

    return(loss,{'p':ap,'n':an,'n_p':nr2h})

#==================================================================================================
#
# MetricLazyQuadrupletLoss
#
#==================================================================================================

class MetricLazyQuadrupletLoss():
  def __init__(self, metric= 'L2', margin1 = 0.5 ,margin2 = 0.5 , alpha = 1, version = 'v2', eps=1e-8, **argv):

    #assert isinstance(margin,list) 
    #assert len(margin) == 2,'margin has to have 2 elements'
    self.reduction = 'mean'
    self.margin1 =  margin1 
    self.margin2 = margin2
    self.metric = metric
    self.eps = torch.tensor(eps)
    self.alpha = alpha
    self.version = version
    # Loss types
    self.LQL = LazyQuadrupletLoss()

    self.loss = get_distance_function(metric)
  
  def __str__(self):
    return type(self).__name__ + ' ' + self.metric

  def __call__(self,descriptor,poses):
    
    # parse pose
    a_pose,p_pose,n_pose = poses['a'],poses['p'],poses['n']
    # parse descriptors
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']

    assert a.shape[0] == 1
    assert p.shape[0] == 1,'positives samples must be 1'
    assert n.shape[0] >= 2,'negative samples must be at least 2' 

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)

    # Descriptors
    #----------------------------------------
    # Anchor - positive
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2)
    # Anchor - negative
    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    # Hard negative
    n_hard_idx = [torch.argmin(neg_loss_array).cpu().numpy().tolist()] # get the negative with smallest distance (aka hard)
    an = neg_loss_array[n_hard_idx]
    # Random negative 
    n_negs    = n_torch.shape[0]
    idx_arr   = np.arange(n_negs)
    elig_neg  = np.setxor1d(idx_arr,n_hard_idx) # Remove from the array the hard negative index
    n_rand_idx  = torch.randint(0,elig_neg.shape[0],(1,)).numpy().tolist()
    dn_hard   = n_torch[n_hard_idx] # Hard negative descriptor
    dn_rand   = n_torch[n_rand_idx] # random negative descriptor
    nr2h= self.loss(dn_hard,dn_rand,dim=2) # distance between hard and random 

    #nn_prime= self.loss(n_torch[elig_neg],dn_rand,dim=2) # among the negatives select subset of eligibles, and compute the distance between NR and all negatives  
    #n_random_hard_idx = [torch.argmin(nn_prime).cpu().numpy().tolist()] # get the negative with smallest distance w.r.t NR (aka NRH)
    #nr2h = nn_prime[n_random_hard_idx] # get the smallest distance between NR and NRH

    # Poses
    #----------------------------------------
    pa_torch,pp_torch  = totensorformat(a_pose,p_pose)
    pap = L2_loss(pa_torch,pp_torch,dim=2)
    pap = torch.__dict__[self.reduction](pap)

    pa_torch,pn_torch  = totensorformat(a_pose,n_pose)
    pan = L2_loss(pa_torch,pn_torch,dim=2)
    pan = torch.__dict__[self.reduction](pan)

    # Compute Loss terms
    #----------------------------------------
    # Compute first term
    s1 = (ap.squeeze()) - an.squeeze() + self.margin1
    first_term = torch.max(self.eps,s1).clamp(min=0.0)
    # Compute second term
    s2 = ap.squeeze() - nr2h.squeeze() + self.margin2
    second_term = torch.max(self.eps,s2).clamp(min=0.0)
    
    if self.version == 'v1':
        third_term =  pap
    elif self.version == 'v2':
        third_term= torch.max(self.eps,(ap/pap)-1)
    else:
        raise ValueError

    loss = first_term + second_term + self.alpha*third_term

    #return(lossv2,info)
    return(loss,{'p':ap,'n':an,'n_p':nr2h,'metric':third_term})

#==================================================================================================
#
#
#==================================================================================================

class LazyTripletplus():
    def __init__(self, metric= 'kl_divergence', margin=0.2, alpha=0.1, version = 'v5',**argv ):
        #self.device= metric
        self.margin = margin
        self.metric = metric
        self.eps    = torch.tensor(1e-8)
        self.alpha  = alpha
        self.reduction = 'mean'
        self.version   = version

        self.loss = get_distance_function(metric) 
    
    def __str__(self):
        return type(self).__name__ + '_' + self.metric + '_' + self.version + '_' +str(self.alpha)

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
        n_pose = n_pose.transpose(1,0)
        p_pose = p_pose.transpose(1,0)
        
        pa_torch,pp_torch  = totensorformat(a_pose,p_pose)
        pap = L2_loss(pa_torch,pp_torch,dim=2)
        pap = torch.__dict__[self.reduction](pap)

        pa_torch,pn_torch  = totensorformat(a_pose,n_pose)
        pan = L2_loss(pa_torch,pn_torch,dim=2)
        pan = torch.__dict__[self.reduction](pan)

        a_torch,p_torch  = totensorformat(a,p)
        pos_loss_array = self.loss(a_torch,p_torch,dim=2)
        fap = torch.__dict__[self.reduction](pos_loss_array)

        a_torch,n_torch = totensorformat(a,n)
        neg_loss_array = self.loss(a_torch, n_torch,dim=2)
        fan = torch.__dict__[self.reduction](neg_loss_array)

        # 1st version
        delta_ap = torch.abs(pap - fap)
        delta_an = torch.abs(pan - fan)
        # 2nd version
        delta_p = torch.abs(pan-pap)
        #delta_an = torch.abs(pan - fan)
        
        if self.version == 'v1':
            l = fap
        elif self.version == 'v2':
            l = fap + fan  
        elif self.version == 'v3':
            l = fap - fan  + self.margin
        elif self.version == 'v4':
            l = fap - fan  + self.margin
        elif self.version == 'v5':
            l =  self.alpha*(delta_ap) + fap # (1-self.alpha)*(delta_ap))
        elif self.version == 'v6':
            l =  self.alpha*(delta_an) + fap # (1-self.alpha)*(delta_ap))
        elif self.version == 'v7':
            l =  self.alpha*(delta_an) + delta_ap # (1-self.alpha)*(delta_ap))
        
        # Added after first results
        elif self.version == 'v8':
            l =  self.alpha*(delta_ap) + (1-self.alpha)*fap
        elif self.version == 'v9':
             l =   fap - fan + delta_p  
        elif self.version == 'v10':
            l = F.mse_loss(fan,pan) + F.mse_loss(fap,pap)
        #elif self.version == 'v9':
        #    l =  self.alpha*(delta_an) + (1-self.alpha)*fap

        value = torch.max(self.eps,l)
        loss = value.clamp(min=0.0)

        return(loss,{'p':fap,'n':delta_an})



#==================================================================================================
#
# Minimal Spanning Tree Matching Loss
#
#==================================================================================================


def kernel_product(w, x, dim=1,distance = 'L2', mode = "gaussian", s = 0.1):
    w_i = w
    x_j = x

    if distance == 'L2':
        xmy = ((w_i - x_j)**2).sum(dim)
    #st()
    if   mode == "gaussian" : K = torch.exp( - (xmy**2 / (s**2) ))
    elif mode == "laplace"  : K = torch.exp( - torch.sqrt(xmy + (s**2)))
    elif mode == "energy"   : K = torch.pow(   xmy + (s**2), -.25 )
    else:
        K = xmy

    return K


def comp_adjacency_matrix(feat_vector,mode = None,distance = 'L2',dim=0):
    n_nodes = feat_vector.shape[0]
    M = torch.zeros((n_nodes,n_nodes))
    for i,a in enumerate(feat_vector):
        for j,b in enumerate(feat_vector):
            M[i,j]  = L2_loss(a,b)
            #M[i,j] = kernel_product(a, b, mode = mode,distance = distance, s = 0.1,dim=dim)
            if i==j:
                M[i,j]=0
    norm_M = F.softmax(M,dim=1)
    return(norm_M)
    

def comp_min_spanning_tree(adjacency_matrix):
    '''
    adjacency_matrix -> [nxn] ,where n are the number of nodes
    '''
    n_nodes = adjacency_matrix.shape[0]
    min_spanning_tree = torch.zeros((n_nodes,n_nodes))

    not_excludes = torch.arange(1,n_nodes)
    curr_node = 0
    for i in range(n_nodes-1):
        
        row = adjacency_matrix[curr_node,not_excludes] # get all nodes that were not yet selected
        next_node = not_excludes[torch.argmin(row).item()] # get global indice
        min_spanning_tree[curr_node,next_node] = 1  # populate matrix with 1, which means there exist a link
        curr_node = next_node   # update current node
        not_excludes = np.setxor1d(not_excludes,next_node) # remove current node from the graph
    return(min_spanning_tree)


class MSTMatchLoss():
    def __init__(self, reduction = 'mean',distance = 'L2',kernel = 'laplace',**argv):
        # self.loss_function = nn.BCELoss()
        self.loss_function = L2_loss
        self.distance = distance
        self.kernel = kernel
        self.reduction = reduction

    def __str__(self):
        return type(self).__name__ + '_' + self.distance
    
    def __call__(self,descriptor,poses):
        pa,pp,pn = poses['a'][0], poses['p'].squeeze(),poses['n'].squeeze()
        vector  = torch.concat((descriptor['a'],descriptor['p']),dim=0)
        poses  = torch.concat((pa,pp),dim=0)
        MP = comp_adjacency_matrix(poses,mode=None,dim=0)
        #print(MP)
        target = comp_min_spanning_tree(MP)
        #print(target)
        input = 1-comp_adjacency_matrix(vector,mode=None,dim=0)
        #print(input)
        #print(input.sum(1))
        value = self.loss_function(input,target)
        value = torch.__dict__[self.reduction](value)
        return value,{}
