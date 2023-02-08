import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.viz import myplot
import imageio
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm import tqdm
import yaml
from dataloader.ORCHARDS import OrchardDataset
import dataloader.ORCHARDS as ORCHARDS 
#from scipy.spatial import distanc


def viz_overlap(xy,
                gt_loops,
                warmupitrs= 10, # Number of frames to ignore at the beguinning
                plot_flag = True,
                record_gif= False, 
                file_name = 'anchor_positive_pair.gif'):

    indices = np.array(range(xy.shape[0]-1))

    if plot_flag == True:
        mplot = myplot(delay=0.5)
        mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'whitesmoke')
        mplot.xlabel('m')
        mplot.ylabel('m')
        
        if record_gif == True:
            mplot.record_gif(file_name)
    

    ROI = indices[2:]
    positives= []
    anchors = []
    for i in ROI:
   
        
        idx = gt_loops[i]
        if len(idx)>0:
            print(len(idx))
            positives.extend(idx)
            anchors.append(i)
        # Generate a colorize the head
        color = np.array(['k' for ii in range(0,i+1)])
        scale = np.array([30 for ii in range(0,i+1)])
        # Colorize the N smallest samples
        color[anchors] = 'r'
        color[positives] = 'b'
        
        scale[positives] = 100
        #color[i] = 'r'
        #print(f'{i}->{sort_}')
        if plot_flag == True and i % 50 == 0:
            
            x = xy[:i,0]#[::-1]
            y = xy[:i,1]#[::-1]
            color = color#[::-1]
            mplot.update_plot(x,y,offset=2,color=color,zoom=0,scale=scale) 






def viz_triplet(pose,triplet_file,record_gif= False):
    anchor, positive, negative = parse_triplet_file(triplet_file)

    plot = myplot(delay = 0.001)
    plot.init_plot(pose[:,0],pose[:,1],c='k',s=10)
    plot.xlabel('m')
    plot.ylabel('m')


    if record_gif == True:
        plot.record_gif('training_data.gif')
        
    for a,p,n in zip(anchor,positive,negative):
        
        c = np.array(['k']*len(pose[:,0]))
        s = np.ones(len(pose[:,0]))*30
        print(f'{str(a-1)} -> {str(p-1)}')
        
        c[a] = 'y'
        c[p] = 'g'
        c[n] = 'r'

        s[a] = 80
        s[p] = 200
        s[n] = 80

        plot.update_plot(pose[:,0],pose[:,1],color=c ,offset= 1, zoom=-1,scale=s)
        plt.pause(0.001)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research-projects/orchards-uk/dataset')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = '',
                                    #default = 'orchard-fr',
                                    type=str,
                                    help=' dataset root directory .'
                                    )
    parser.add_argument('--seq',    
                                    default  = 'autumn',
                                    type = str)
    parser.add_argument('--plot',default  = True ,type = bool)
    parser.add_argument('--loop_thresh',default  = 1 ,type = float)
    parser.add_argument('--record_gif',default  = False ,type = bool)
    parser.add_argument('--cfg',default  = 'cfg/overlap_cfg.yaml' ,type = str)
    parser.add_argument('--option',default  = 'compt' ,type = str,choices=['viz','compt'])
    parser.add_argument('--pose_file',default  = 'poses' ,type = str)
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot
    record_gif_flag = args.record_gif
    cfg_file = args.cfg
    option = args.option
    loop_thresh = args.loop_thresh

    print("[INF] Dataset Name:  " + dataset)
    print("[INF] Sequence Name: " + seq )
    print("[INF] Plotting Flag: " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    print("[INF] cfg: " + cfg_file)
    print("[INF] Reading poses from : " + args.pose_file)

    ground_truth = {'pos_range':15, # Loop Threshold [m]
                    'neg_range': 17,
                    'num_neg':20,
                    'num_pos':50,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500}
    
    dataset = OrchardDataset(root,'',seq,sync = True,ground_truth=ground_truth)
    
    table = dataset._get_gt_()
    true_loop = np.array([np.where(line==1)[0] for line in table])
    
    xy = dataset._get_pose_()
    n_point = xy.shape[0]
    

    viz_overlap(xy,true_loop)

 


    

