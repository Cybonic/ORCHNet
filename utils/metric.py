

import numpy as np



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


def retrieve_eval(retrieved_map,true_relevant_map,top=1,**argv):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert top > 0
  n_queries = retrieved_map.shape[0]
  precision, recall = 0,0
  for retrieved,relevant in zip(retrieved_map,true_relevant_map):
    top_retrieved = retrieved[:top] # retrieved frames for a given query
    
    tp = 0 # Reset 
    if any(([True  if cand in relevant else False for cand in top_retrieved])):
        # It is only required to find one loop per anchor in a set of retrieved frames
        tp = 1 
    
    recall += tp # recall = tp/1
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r.t the query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})




def retrieve_metricsv2(relevant_hat,true_relevant,top=1,mode = 'hard'):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert mode in ['relaxe', 'hard']

  n_queries = relevant_hat.shape[0]
  precision, recall = 0,0
  for p,g in zip(relevant_hat,true_relevant):
    top_relevant_hat = p[:top]

    n_true_loops = 1 if mode =='relaxe' else len(g) # Number of loops per anchor = 1
    
    num_tp = np.sum([1 for c in top_relevant_hat if c in g]) # Number of predicted loops
    tp  = 1 if mode == 'relaxe' and num_tp > 0 else num_tp  # In "relaxe" mode, 
    
    recall +=tp/n_true_loops #  (Recall w.r. to query) retrieved loops/ universe of loops 
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r. to query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})