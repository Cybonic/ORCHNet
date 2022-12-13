from fnmatch import fnmatch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from  tqdm import tqdm 
import torch
import utils.loss as loss_lib
from .loss import cosine_loss,totensorformat 

class knn_retrieval():
    def __init__(self,metric, top_cand, range = None):
        self.range_flag = range

        if self.range_flag != None: 
            self.neigh = NearestNeighbors(metric = metric, radius = range , p = 2 )
        elif top_cand != None:
            self.neigh = NearestNeighbors(metric = metric, n_neighbors = top_cand , p = 2 )
        else:
            raise ValueError("No value defined for 'top_cand' or 'range'")
    
    def fit(self,input):
        self.neigh.fit(input)
    
    def k_inference(self,input):
        
        #input = input.reshape(-1, 1) # required for Kneighbors
        if self.range_flag != None: 
            score,nn = self.neigh.radius_neighbors(input)
        else:
            score,nn = self.neigh.kneighbors(input)

        return(score,nn)



def place_knn(descriptors_dict, top_cand = 1,sim_thres = 0.5, text = '', burn_in = 50):
    '''
    
    
    '''
    num_dscptor = len(descriptors_dict)

    retrieval = knn_retrieval(metric='euclidean',range = None, top_cand = top_cand)
    pred_map = np.zeros((num_dscptor,num_dscptor),np.uint8)

    descriptors  = np.array(list(descriptors_dict.values()))
    global_index = np.array(list(descriptors_dict.keys()))

    tamplate = (np.ones((1,top_cand),np.int32)*-1)[0]

    top_scores_array = (np.ones((burn_in,top_cand),np.int32)*-1).tolist()
    top_cand_array   = (np.ones((burn_in,top_cand),np.int32)*-1).tolist()

    base_idx = np.arange(stop = descriptors.shape[0])
    for i in tqdm(range(burn_in,num_dscptor),text):
        
        j = (i-burn_in)
        if j < top_cand:
            top_scores_array.append(tamplate.tolist())
            top_cand_array.append(tamplate.tolist())
            continue

        base_idx_  = base_idx[base_idx<=j]
        dsctor_map = descriptors[base_idx_]
        
        retrieval.fit(dsctor_map)
        query = np.array(descriptors[i])
        # Fit map
        # Skip until burn-in 
        scores,winners = retrieval.k_inference(query.reshape(1,-1))
        #  only consider as true if score > similarity threshold 
        scores = scores[0]
        winners = global_index[base_idx_[winners[0]]]

        roi  = scores<sim_thres

        if roi.all() == False:
            w_tamplate = tamplate.copy()
            s_tamplate = tamplate.copy()
        else:
            w_tamplate = tamplate.copy()[roi]= winners[roi]
            s_tamplate = tamplate.copy()[roi]= scores[roi]

        #if scores < sim_thres:
        top_cand_array.append(w_tamplate.tolist())
        top_scores_array.append(s_tamplate.tolist())
    
    return np.array(top_cand_array,dtype= np.int32), np.array(top_scores_array)




def sim_knn(query,map,top_cand,metric='cosine_loss'):
    metric_fun = loss_lib.__dict__[metric]
    scores,winner = [],[]
    for q in query:
        q_torch,map_torch = totensorformat(q.reshape(1,-1),map) 
        sim = metric_fun(q_torch,map_torch) # similarity-based metrics 0 :-> same; +inf: -> Dissimilar 
        sort_value,sort_idx = sim.sort() # Sort to get the most similar vectors first
        # save top candidates
        scores.append(sort_value.detach().cpu().numpy()[:top_cand])
        winner.append(sort_idx.detach().cpu().numpy()[:top_cand])

    return np.array(winner),np.array(scores)




def sim_relocalize(descriptors_dict, top_cand = 1,sim_thres = 0.5, text = '', burn_in = 50):
    '''
    
    
    '''
    num_dscptor = len(descriptors_dict)

    #retrieval = knn_retrieval(metric='euclidean',range = None, top_cand = top_cand)
    pred_map = np.zeros((num_dscptor,num_dscptor),np.uint8)

    descriptors  = np.array(list(descriptors_dict.values()))
    global_index = np.array(list(descriptors_dict.keys()))

    tamplate = (np.ones((1,top_cand),np.int32)*-1)[0]

    top_scores_array = (np.ones((burn_in,top_cand),np.int32)*-1).tolist()
    top_cand_array   = (np.ones((burn_in,top_cand),np.int32)*-1).tolist()

    base_idx = np.arange(stop = descriptors.shape[0])
    for i in tqdm(range(burn_in,num_dscptor),text):
        
        j = (i-burn_in)
        if j < top_cand:
            top_scores_array.append(tamplate.tolist())
            top_cand_array.append(tamplate.tolist())
            continue

        base_idx_  = base_idx[base_idx<=j]
        dsctor_map = descriptors[base_idx_]
        
        #retrieval.fit(dsctor_map)
        query = np.array(descriptors[i])
        # Fit map
        # Skip until burn-in 
        scores,winners = sim_knn(query.reshape(1,-1),dsctor_map,top_cand=top_cand)
        #  only consider as true if score > similarity threshold 
        scores = scores[0]
        winners = global_index[base_idx_[winners[0]]]

        roi  = scores<sim_thres

        if roi.all() == False:
            w_tamplate = tamplate.copy()
            s_tamplate = tamplate.copy()
        else:
            w_tamplate = tamplate.copy()[roi]= winners[roi]
            s_tamplate = tamplate.copy()[roi]= scores[roi]

        #if scores < sim_thres:
        top_cand_array.append(w_tamplate.tolist())
        top_scores_array.append(s_tamplate.tolist())
    
    return np.array(top_cand_array,dtype= np.int32), np.array(top_scores_array)




def euclidean_knnv2(anchors, map, top_cand = 10,range_value = None, text=''):
    '''
    
    '''
    retrieval = knn_retrieval(metric='euclidean',range = range_value, top_cand = top_cand)
    retrieval.fit(map)

    scores,winners = retrieval.k_inference(anchors)

    return(winners,scores)




def retrieval_knn(query_dptrs,map_dptrs, top_cand,metric):
    
    #retrieved_loops ,scores = euclidean_knnv2(query_dptrs,map_dptrs, top_cand= max_top)
    metric_fun = loss_lib.get_distance_function(metric)
    scores,winner = [],[]

    for q in query_dptrs:
        q_torch,map_torch = totensorformat(q.reshape(1,-1),map_dptrs) 
        sim = metric_fun(q_torch,map_torch,dim=2).squeeze() # similarity-based metrics 0 :-> same; +inf: -> Dissimilar 
        sort_value,sort_idx = sim.sort() # Sort to get the most similar vectors first
        # save top candidates
        scores.append(sort_value.detach().cpu().numpy()[:top_cand])
        winner.append(sort_idx.detach().cpu().numpy()[:top_cand])

    return np.array(winner),np.array(scores)

    # retrieved_loops ,scores = sim_knn(query_dptrs,map_dptrs, top_cand = max_top,metric='cosine_loss')




def comp_queries_score_table(target,queries):
    '''
    
    '''
    if not isinstance(target,np.ndarray):
        target = np.array(target)
    
    if not isinstance(queries,np.ndarray):
        queries = np.array(queries)

    table_width = target.shape[0]
    idx = np.arange(table_width)
    target_wout_queries = np.setxor1d(idx,queries)
    
    table = np.zeros((queries.shape[0],target_wout_queries.shape[0]),dtype=np.float32)
    
    for i,(q) in enumerate(queries):
        qdistance = np.linalg.norm(target[q,:]-target[target_wout_queries,:],axis=1)
        table[i,:]= qdistance
    
    table_struct = {'idx':target_wout_queries,'table':table}
    return(table_struct)



def comp_score_table(target):
    '''
    
    '''
    if not isinstance(target,np.ndarray):
        target = np.array(target)
    
    table_width = target.shape[0]
    
    table = np.zeros((table_width,table_width),dtype=np.float32)
    table = []
    for i in range(table_width):
        qdistance = np.linalg.norm(target[i,:]-target,axis=1)
        table.append(qdistance.tolist())
    return(np.asarray(table))




def gen_ground_truth(pose,anchor,pos_thres,neg_thres,num_neg,num_pos):
    '''
    input args: 
        pose [nx3] (x,y,z)
        anchor [m] indices
        pos_thres (scalar): max range of positive samples 
        neg_thres (scalar): min range of negative samples 
        num_neg (scalar): number of negative sample to return
        num_pos (scalar): number of positive sample to return
    
    return 
        positive indices wrt poses
        negative indices wrt poses
    '''
    # assert mode in ['hard','distribution']

    table  = comp_score_table(pose)

    all_idx = np.arange(table.shape[0])
    wout_query_idx = np.setxor1d(all_idx,anchor)
    positive = []
    negative = []
    
    for a in zip(anchor):

        query_dist = table[a]
        #selected_idx = np.where(query_dist>0)[0] # exclude anchor idx (dist = 0)
        #sort_query_dist_idx  =  np.argsort(query_dist)
        all_pos_idx  = np.where(query_dist < pos_thres)[0]
        sort_pos_idx = np.argsort(query_dist[all_pos_idx])
        sort_all_pos_idx = all_pos_idx[sort_pos_idx]
        
        dis = query_dist[sort_all_pos_idx]
        all_pos_idx  = sort_all_pos_idx[1:] # remove the 0 element 
     
        dis_top = query_dist[all_pos_idx]

        tp  = np.array([i for i in all_pos_idx if i not in anchor])
        #tp = np.setxor1d(all_pos_idx,anchor)
        if len(tp)>num_pos and num_pos>0:
            tp = tp[:num_pos]
            dis_top = query_dist[tp]
            #pos_idx = np.random.randint(low=0,high = len(tp),size=num_pos)
            #tp = tp[pos_idx]

        all_neg_idx = np.where(query_dist>neg_thres)[0]
        neg_idx = np.random.randint(low=0,high = len(all_neg_idx),size=num_neg)
        tn = all_neg_idx[neg_idx]

            
            #neg_dists = np.argsort(query_dist[all_neg_idx])
            #dd = query_dist[all_neg_idx][neg_dists]
            #neg_idx = neg_dists[:num_neg]

        #elif mode == 'hard':
        #    tp = [selected_idx[np.argmin(query_dist)]]
        #    all_neg_idx= np.where(query_dist>neg_thres)[0]
        #    tn = [selected_idx[all_neg_idx[np.argmin(query_dist[all_neg_idx])]]]
        
        positive.append(tp)
        negative.append(tn)

    return(np.array(positive),np.array(negative))



def comp_gt_table(pose,anchors,pos_thres):
    '''
    
    
    '''
    table  = comp_score_table(pose)
    num_pose = pose.shape[0]
    gt_table = np.zeros((num_pose,num_pose),dtype=np.uint8)
    all_idx  = np.arange(table.shape[0])
    idx_wout_anchors = np.setxor1d(all_idx,anchors) # map idx: ie all idx excep anchors

    for anchor in anchors:
        anchor_dist = table[anchor]
        all_pos_idx = np.where(anchor_dist < pos_thres)[0] # Get all idx on the map that form a loop (ie dist < thresh)
        tp = np.intersect1d(idx_wout_anchors,all_pos_idx).astype(np.uint32) # Remove those indices that belong to the anchor set
        gt_table[anchor,tp] = 1 # populate the table with the true positives

    return(gt_table)




def evaluation(relevant_hat,true_relevant, top=1,mode = 'relaxe'):
    '''
    https://amitness.com/2020/08/information-retrieval-evaluation/
    
    '''
    return  
    
    #return 



def evaluationv2(pred,gt,type = 'hard',smooth=0.000001):
    '''
    https://amitness.com/2020/08/information-retrieval-evaluation/
    
    '''
    P = np.sum(gt == 1)
    N = np.sum(gt == 0)

    num_samples = pred.shape[0]
    
    eval_rate = []
    for pred_line,gt_line in zip(pred,gt):
        tp,fp,tn,fn = 0,0,0,0
        if np.sum(gt_line == 1)>0:
            tp = 1 if np.sum(((gt_line == 1) & (pred_line==1))==1) > 0 else 0
            if tp ==0:
                fn = 1 if np.sum(((gt_line == 1) & (pred_line==0))==1) > 0 else 0
        else:
            positives = np.sum(pred_line == 1)
            fp = 1 if positives > 0  else 0
            tn = 1 if positives == 0 else 0
        eval_rate.append([tp,fp,tn,fn])

    rate = np.sum(eval_rate,axis=0)
    tp,fp,tn,fn = rate[0],rate[1],rate[2],rate[3]

    return tp,fp,tn,fn 



def retrieval_metric(tp,fp,tn,fn):

    b = tp+tn+fp+fn  
    accuracy = (tp+tn)/b if b!=0 else 0

    b = tp+fn
    recall = tp/b if b!=0 else 0

    b = tp+fp
    precision = tp/b if b!=0 else 0
       
    b = precision + recall 
    F1 = 2*((precision*recall) /(precision + recall)) if b!=0 else 0

    return({'recall':recall, 'precision':precision,'F1':F1,'accuracy':accuracy})
