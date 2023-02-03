import os
from .utils import *
from .laserscan import LaserData
from PIL import Image 
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as Tr
from  torch.utils.data.sampler import SubsetRandomSampler
import yaml
import numpy as np
import tqdm
import torch


ASEGMENTS = [   {'xmin':-15,'xmax':-9,'ymin':-49,'ymax':-3 },
                {'xmin':-9,'xmax':-5,'ymin':-49,'ymax':-3 },
                {'xmin':-5,'xmax':-2,'ymin':-49,'ymax':-3 },
                {'xmin':-2,'xmax':2,'ymin':-49,'ymax':-3 },
                {'xmin':-15,'xmax':2,'ymin':-55,'ymax':-49 },
                {'xmin':-15,'xmax':2,'ymin':-3,'ymax':5 }
                ]


PREPROCESSING = Tr.Compose([Tr.ToTensor()])

MODALITIES = ['range','projection','remissions','mask']

def subsampler(universe,num_sub_samples):
        assert  len(universe)>num_sub_samples
        return np.random.randint(0,len(universe),size=num_sub_samples)



def gen_ground_truth(   poses, 
                        pos_range= 0.05, # Loop Threshold [m]
                        neg_range=10,
                        num_neg = 10,
                        num_pos = 10,
                        warmupitrs= 10, # Number of frames to ignore at the beguinning
                        roi       = 5 # Window):
                    ):

    indices = np.array(range(poses.shape[0]-1))
    loop_labels = np.zeros(poses.shape[0],dtype=int)
    

    ROI = indices[warmupitrs:]
    anchor =   []
    positive = []
    select_pos_idx = np.arange(num_pos)
    for i in ROI:
    
        _map_   = poses[:i,:]
        pose    = poses[i,:].reshape((1,2))
        map_frame_idx  = indices[:i]
        
        dist_meter = np.linalg.norm(pose-_map_,axis=1)

        #dist= np.sqrt((pose[0]-_map_[:,0])**2 + (pose[1]-_map_[:,1])**2)
        dist = dist_meter/ np.max(dist_meter)

        frame_dist = np.linalg.norm(i-map_frame_idx)
        #frame_dist= np.sqrt((i-map_frame_idx)**2 + (i-map_frame_idx)**2)
        frame_dist /= np.max(frame_dist)

        alpha = dist/frame_dist
        # Sort distance and get smallest  outside ROI 
        sort_=np.argsort(alpha[:i-roi])
        pos_sort_ = sort_[select_pos_idx]
        pos_idx = np.where(dist_meter[pos_sort_] < pos_range)[0]
        
        # Select only those positives that are in the same line then the anchor
        #an_seg = get_roi_points(pose,ASEGMENTS)
        #print(an_seg)
        #pos_seg = get_roi_points(poses[pos_idx,:],ASEGMENTS)
        #pos_idx_ref = [idx for idx, pos in zip(pos_idx,pos_seg) if pos in an_seg]
        #print(pos_seg)
        # save
        if  len(pos_idx)>0:
            anchor.append(i)
            positive.append([sort_[idx] for idx in pos_idx])
        else: 
            sort_=-1
    
    # Negatives
    negatives= []
    neg_idx = np.arange(num_neg)   
    for a, pos in zip(anchor,positive):
        pa = poses[a,:].reshape((1,2))
        dist_meter = np.linalg.norm(pa-poses,axis=1)
        neg_idx = np.where(dist_meter > neg_range)[0]
        neg_idx = np.setxor1d(neg_idx,pos)
        select_neg = np.random.randint(0,len(neg_idx),num_neg)
        neg_idx = neg_idx[select_neg]
        negatives.append(neg_idx)

    return(anchor,positive,negatives)


def get_roi_points(points,rois):
    points = points.reshape((-1,2))
    labels = []
    for j,point in enumerate(points):
        pointlabel = []
        for i,roi in enumerate(rois):
            roi_dx = ((point[0]>=roi['xmin']).astype(np.int8) * (point[0]<roi['xmax']).astype(np.int8) *
                      (point[1]>=roi['ymin']).astype(np.int8) * (point[1]<roi['ymax']).astype(np.int8))
            if roi_dx == 1:
                pointlabel=i
                break
        labels.append(pointlabel)   
    return labels


def get_point_cloud_files(dir):
    path = dir
    if not os.path.isdir(path):
        print("[ERR] Path does not exist!")
    files = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return(files)




class parser():
    def __init__(self):
        self.dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2'), ('time', '<f4')]

    def velo_read(self,file):
        input_file = open(file,'rb')
        scan = np.fromfile(input_file, dtype=self.dt)
        scan = np.hstack((scan['x'].reshape((-1,1)),scan['y'].reshape((-1,1)),scan['z'].reshape((-1,1)),scan['intensity'].reshape((-1,1))))
        return scan



# ========================================================================================================

class OrchardDataset():
    def __init__(self,
                    root,
                    dataset,
                    seq,
                    sync = True , 
                    modality = 'pcl' ,
                    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                                     'neg_range': 10,
                                     'num_neg':20,
                                     'num_pos':1,
                                     'warmupitrs': 600, # Number of frames to ignore at the beguinning
                                     'roi':500},
                        **argv):

        self.modality = modality

        # Load dataset and laser settings
        cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))

        if not 'square_roi' in argv:
            argv['square_roi'] = [sensor_cfg[seq]['roi']]

        if modality in ['range']:
            self.param = sensor_cfg[seq]['RP']
        elif modality in ['bev']:
            self.param = sensor_cfg[seq]['BEV']
        else:
            self.param = {}

        self.laser = LaserData(
                parser=parser(),
                project=True,
                **argv
                )

        # 
        self.target_dir = os.path.join(root,dataset,seq)
        assert os.path.isdir(self.target_dir),'target dataset does nor exist: ' + self.target_dir

        pose_file = os.path.join(self.target_dir,'poses.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_pose_to_RAM(pose_file)

        point_cloud_dir = os.path.join(self.target_dir,'point_cloud')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        names, self.point_cloud_files = get_files(point_cloud_dir)

        if sync == True:
            sync_plc_idx_file = os.path.join(self.target_dir,'sync_point_cloud_idx.txt')
            assert os.path.isfile(pose_file),'sync plc file does not exist: ' + sync_plc_idx_file

            sync_pose_idx_file = os.path.join(self.target_dir,'sync_poses_idx.txt')
            assert os.path.isfile(pose_file),'sync pose file does not exist: ' + sync_pose_idx_file

            sync_pose_idx = load_indices(sync_pose_idx_file)
            sync_plc_idx = load_indices(sync_plc_idx_file)

            self.point_cloud_files = self.point_cloud_files[sync_plc_idx]
            self.pose =  self.pose[sync_pose_idx]
       
        self.anchors,self.positives,self.negatives = gen_ground_truth(self.pose,**ground_truth)

        n_points = self.pose.shape[0]
        self.table = np.zeros((n_points,n_points))
        for a,p in zip(self.anchors,self.positives):
            self.table[a,p]=1


    def __len__(self):
        return self.pose.shape[0]

    def __call__(self,idx):
        return self.load_point_cloud(idx)

    def _get_gt_(self):
        return self.table

    def _get_pose_(self,idx=np.nan):
        if not isinstance(idx,(np.ndarray, np.generic)):
            idx = np.array(idx)
        if idx.size==1 and np.isnan(idx):
            return(self.pose)
        return(self.pose[idx])
    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files)
        return(self.point_cloud_files[idx])
    
    def load_point_cloud(self,idx):
        file = self.point_cloud_files[idx]
        self.laser.open_scan(file)
        pclt = self.laser.get_pcl()
        return pclt
        
    def _get_modality_(self,idx):
        file = self.point_cloud_files[idx]
        self.laser.open_scan(file)
        pclt = self.laser.get_data(self.modality,self.param)
        return pclt





# ========================================================================================================
# Evaluation dataloader for the second stage 

class ORCHARDSEval(OrchardDataset):
    def __init__(self,root, dataset, sequence, sync = True,   # Projection param and sensor
                modality = 'range' , 
                mode = 'Disk', 
                **argv
                ):
        
        super(ORCHARDSEval,self).__init__(root, dataset, sequence, sync=sync, modality=modality,**argv)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING
        self.line_rois = ASEGMENTS

        self.num_samples = self.point_cloud_files.shape[0]
        self.idx_universe = np.arange(self.num_samples)

        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
        self.poses = self._get_pose_()
   
        assert len(np.intersect1d(self.anchors,self.map_idx)) == 0, 'No indicies should be in both anchors and map'

        if self.mode == 'RAM':
            self.inputs = self.load_RAM()


    def get_GT_Map(self):
        return(self.table)
    
    def get_eval_data(self,index):
        global_index = self.idx_universe[index] # Only useful when subsampler is on
        
        if self.mode == 'RAM':
            data = self.inputs[global_index]         
        elif self.mode == 'Disk':
            data = self._get_modality_(global_index)
        
        plc = self.preprocessing(data)
        return(plc,global_index)
        return data

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_representation_(i)
            img[i]=data#.astype(np.uint8)
        return img


    def __getitem__(self,index):
        pcl,gindex = self.get_eval_data(index)
        return(pcl,gindex)

    def __len__(self):
        return(self.num_samples)
        
    def get_pose(self):
        pose = self._get_pose_()
        return pose

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)
    
    def comp_line_loop_table(self,pose):
        '''
        returns a array with same size of poses. with labels of each line
        '''
        segments = np.zeros(pose.shape[0],dtype=np.uint8)
        for i, roi in enumerate(self.line_rois):
            roi_dx = get_roi_points(pose,roi)
            #print(roi_dx.sum())
            segments[roi_dx] = i 
        return(segments)
    





class ORCHARDSTriplet(OrchardDataset):
    def __init__(self,
                        root,
                        dataset,
                        sequence, 
                        sync = True, 
                        mode='Disk', 
                        modality = 'projection', 
                        aug=False,
                        num_subsamples = 0,
                        **argv):

        super(ORCHARDSTriplet,self).__init__(root,dataset,sequence, sync = sync, modality=modality,**argv)

        self.modality = modality
        self.aug_flag = aug
        self.mode = mode
        self.eval_mode = False
        self.line_rois = ASEGMENTS

        self.preprocessing = PREPROCESSING
        verbose = True
    

        # Triplet data
        self.num_samples = len(self._get_point_cloud_file_())
        self.idx_universe = np.arange(self.num_samples)
        # Eval data
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
        self.poses = self._get_pose_()

        if 'subsample' in argv and argv['subsample'] > 0:
            self.set_subsampler(argv['subsample'])

         # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

        
    def line_wise_triplet_split(self,gt_line_labels,anchors,num_neg,num_pos):
        classes = np.unique(gt_line_labels)
        positives =[]
        negatives = []
        all_idx = np.arange(len(gt_line_labels))
        for anchor in anchors:
            my_line = gt_line_labels[anchor]
            # Positives
            all_positives = np.where(gt_line_labels==my_line)[0]
            positives_idx = all_positives[all_positives!=anchor] # Remove current anchor idx
            positive_random_idx = np.random.randint(0,len(positives_idx),num_pos)
            positives.append(positives_idx[positive_random_idx])
            all_negative_idx  = np.setxor1d(all_idx,positives_idx)
            neg_random_idx = np.random.randint(0,len(all_negative_idx),num_neg)
            negatives.append(all_negative_idx[neg_random_idx])
        
        return np.array(positives),np.array(negatives)




    def set_subsampler(self,percentage):
        num_samples = int(len(self.anchors)*percentage)
        print("Number of samples: " + str(num_samples))
        subset_idx = subsampler(self.anchors,num_samples)
        # Select subsets
        self.anchors =  np.array(self.anchors)[subset_idx]
        self.positive = np.array(self.positive)[subset_idx]
        self.negative = np.array(self.negative)[subset_idx]
        # Filter repeated samples to maintain memory footprint low
        from utils.utils  import unique2D
        positive = unique2D(self.positive) 
        negative = unique2D(self.negative)
        self.idx_universe =np.unique(np.concatenate((self.anchors,positive,negative)))
        self.num_samples = len(self.idx_universe)

       
        
    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_modality_(i)
            img[i]=data#.astype(np.uint8)
        return img
    
    def get_eval_data(self,index):
        global_index = self.idx_universe[index] # Only useful when subsampler is on
        
        if self.mode == 'RAM':
            data = self.inputs[global_index]         
        elif self.mode == 'Disk':
            data = self._get_modality_(global_index)
        
        plc = self.preprocessing(data)
        return(plc,global_index)

    def get_triplet_data(self,index):
        an_idx,pos_idx,neg_idx  = self.anchors[index],self.positives[index], self.negatives[index]
        if self.mode == 'RAM':     
            # point clouds are already converted to the input representation, is only required to 
            #  convert to tensor 
            plt_anchor = self.preprocessing(self.inputs[an_idx])
            plt_pos = torch.stack([self.preprocessing(self.inputs[i]) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self.inputs[i]) for i in neg_idx],axis=0)

        elif self.mode == 'Disk':
            plt_anchor = self.preprocessing(self._get_modality_(an_idx))
            plt_pos = torch.stack([self.preprocessing(self._get_modality_(i)) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self._get_modality_(i)) for i in neg_idx],axis=0)
          
        else:
            raise NameError

        pcl = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':len(plt_anchor),'positive':len(plt_pos),'negative':len(plt_neg)}
        return(pcl,indx)

    def set_eval_mode(self,mode=True):
        self.eval_mode = mode

    def __getitem__(self,index):
        if not self.eval_mode:
            pcl,indx = self.get_triplet_data(index)
        else:
            pcl,indx = self.get_eval_data(index)
        return(pcl,indx)
    
    def __len__(self):
        if not self.eval_mode:
            num_sample = len(self.anchors)
        else:
            num_sample = len(self.idx_universe)
        return(num_sample)
    
    def get_pose(self):
        return self._get_pose_()

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)
    
    def get_GT_Map(self):
        return(self.table)
    
    def comp_line_loop_table(self,pose):
        '''
        returns a array with same size of poses. with labels of each line
        '''
        segments = np.ones(pose.shape[0],dtype=np.int8)*-1
        for i, roi in enumerate(self.line_rois):
            roi_dx = get_roi_points(pose,roi)
            #print(roi_dx.sum())
            #assert all(segments[roi_dx]==-1) 
            segments[roi_dx] = i 
        return(segments)




# ===================================================================================================================
#
#
#
# ===================================================================================================================


class ORCHARDS():
    def __init__(self,train_loader,test_loader, split_mode='cross-val', **kwargs):

        self.valloader = None
        self.trainloader = None

        assert split_mode in ['cross-val','train-test','same'], "Split mode not recognized: " + split_mode
        import copy
        test_set = None
        train_set = ORCHARDSTriplet(root = kwargs['root'],
                                            mode = kwargs['mode'],
                                            **train_loader['data'],
                                            ground_truth = train_loader['ground_truth']
                                            #subsample = 0.5
                                            )

        if split_mode == 'cross-val':
            # Cross-validation. Train and test sets are from different sequences
            
               
            test_set = ORCHARDSEval( root =  kwargs['root'],
                                                mode = kwargs['mode'],
                                                #num_subsamples = num_subsamples,
                                                **test_loader['data'],
                                                ground_truth = test_loader['ground_truth']
                                                )

        elif  split_mode == 'train-test':
            # train-test: train and test sets are from the same sequences, which is split randomly in two.
            # Before
            train_size = 0.7
            print('Train data set:', len(train_set))
            print("train-test split: " + str(train_size))
            # Random split
            train_set_size = int(len(train_set) * train_size)
            valid_set_size = len(train_set) - train_set_size
            train_set, test_set = data.random_split(train_set, [train_set_size, valid_set_size])
            test_set.dataset = copy.copy(test_set.dataset) # Mandatory to guarantee independence from the training
            test_set.dataset.set_eval_mode(True)
            #train_set.dataset.set_eval_mode(False)
        else:
            # Test on the same dataset as trained (Only for debugging)
            test_set =  copy.copy(train_set)
            test_set.set_eval_mode(True)
        
        #print(test_set.indices)
        #print(train_set.indices)

        self.trainloader  = DataLoader(train_set,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = train_loader['shuffle'],
                                    num_workers= 0,
                                    pin_memory=True,
                                    drop_last=True,
                                    )

        self.valloader   = DataLoader(test_set,
                                    batch_size = test_loader['batch_size'],
                                    num_workers= 0,
                                    pin_memory=True,
                                    )

    def get_train_loader(self):
        return self.trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    def get_val_loader(self):
        return  self.valloader
    
    def get_label_distro(self):
        raise NotImplemented
		#return  1-np.array(self.label_disto)
        #   
def conv2PIL(image):
    min_val = np.min(image)
    max_val = np.max(image)
    value = (image-min_val)
    nomr_val = (value/np.max(value))*255
    im_pil = Image.fromarray(nomr_val)
    im_pil = im_pil.convert("L")
    return(im_pil)

   


