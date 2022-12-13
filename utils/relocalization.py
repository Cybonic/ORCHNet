from fnmatch import fnmatch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from  tqdm import tqdm 
import torch
import utils.loss as loss_lib
from .loss import cosine_loss,totensorformat 






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








def relocal_metric(relevant_hat,true_relevant):
    '''
    Difference between relocal metric and retrieval metric is that 
    retrieval proseposes that only evalautes positive queries
    ...

    input: 
    - relevant_hat (p^): indices of 
    '''
    n_samples = len(relevant_hat)
    recall,precision = 0,0
    
    for p,g in zip(relevant_hat,true_relevant):
        p = np.array(p).tolist()
        n_true_pos = len(g)
        n_pos_hat = len(p)
        tp = 0 
        fp = 0
        if n_true_pos > 0: # postive 
            # Loops exist, we want to know if it can retireve the correct frame
            num_tp = np.sum([1 for c in p if c in g])
            
            if num_tp>0: # found at least one loop 
                tp=1
            else:
                fn=1 
            
        else: # Negative
            # Loop does not exist: we want to know if it can retrieve a frame 
            # with a similarity > thresh
            if n_pos_hat == 0:
                tp=1

        recall += tp/1 
        precision += tp/n_pos_hat if n_pos_hat > 0 else 0
    
    recall/=n_samples
    precision/=n_samples
    return {'recall':recall, 'precision':precision}
