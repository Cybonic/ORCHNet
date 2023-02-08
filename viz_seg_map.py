from dataloader.ORCHARDS import *

import yaml
import matplotlib.pyplot as plt

from utils.viz import myplot


if __name__=='__main__':
    session = 'orchard-uk.yaml'
    session_cfg_file = os.path.join('sessions', session)
 
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    root = SESSION['root']
    sequence = 'autumn'
    
    ground_truth = {
        'pos_range': 10, # Loop Threshold [m]
        'neg_range': 17,
        'num_neg': 20,
        'num_pos': 50,
        'warmupitrs': 600, # Number of frames to ignore at the beguinning
        'roi': 500
     }
    loader = ORCHARDSEval(root=root,dataset='',sequence=sequence,sync=True,modality='pcl',ground_truth=ground_truth)

    poses  = loader.get_pose()
    anchor = loader.get_anchor_idx()
    map    = loader.get_map_idx()
    table  = loader.get_GT_Map() 

    true_loop = np.array([np.where(line==1)[0] for line in table])

    positives = true_loop[anchor]

    mplot = myplot(delay=0.5)
    mplot.init_plot(poses[:,0],poses[:,1],s = 10, c = 'whitesmoke')
    mplot.xlabel('m')
    mplot.ylabel('m')

    # gt_line_labels = loader.get_GT_Map()
    c = np.array(['r','b','y','k','g','m'])
    color = np.array(['whitesmoke']*poses.shape[0])
    scale = np.ones(poses.shape[0])*10

    indices = np.array(range(poses.shape[0]-1))
    #positives = []
    #anchors = []
    ROI = indices[2:]
    pos_range = 10
    roi = 100
    # ASEGMENTS = ORCHARDS.ASEGMENTS
    for  i in indices[roi+2:]:
        
        _map_   = poses[:i,:]
        pose    = poses[i,:].reshape((1,-1))
        #dist_meter  = np.sqrt(np.sum((pose -_map_)**2,axis=1))
       
        #pos_idx = np.where(dist_meter[:i-roi] < pos_range)[0]
        anchors = []
        positives = []
        pos_idx = true_loop[i]
        #pos = true_loop[i]
        if len(pos_idx)>0:
            # anchors=i
            # positives = pos_idx
            #print(len(pos))
            
            pa = poses[i].reshape(-1,2)
            pp = poses[pos_idx].reshape(-1,2)
            
            an_labels, an_point_idx = get_roi_points(pa,ASEGMENTS)
            #alabel = list(line_paa.keys())
            pos_labels, pos_point_idx = get_roi_points(pp,ASEGMENTS)
            # plabel = np.array(list(line_ppa.keys()))

            boolean_sg = np.where(an_labels[0] == pos_labels)[0]
            if len(boolean_sg):
                positives = [pos_idx[pos_point_idx[idx]] for idx in boolean_sg]
                anchors.append(i)


        # Generate a colorize the head
        color = np.array(['k' for ii in range(0,i+1)])
        scale = np.array([30 for ii in range(0,i+1)])
        # Colorize the N smallest samples
        
        color[positives] = 'b'
        color[anchors] = 'r'
        scale[positives] = 100

        if i % 20 == 0:
            mplot.update_plot(poses[:i+1,0],poses[:i+1,1],offset=2,zoom=0,color=color,scale=scale)









