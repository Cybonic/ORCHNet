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
from tqdm import tqdm
import torch


AUTUMN = [   {'xmin':-15,'xmax':-9,'ymin':-50,'ymax':-1 },
                {'xmin':-9,'xmax':-5,'ymin':-50,'ymax':-1 },
                {'xmin':-5,'xmax':-2,'ymin':-50,'ymax':-1 },
                {'xmin':-2,'xmax':2,'ymin':-50,'ymax':-1 },
                {'xmin':-15,'xmax':2,'ymin':-55,'ymax':-49 },
                {'xmin':-15,'xmax':2,'ymin':-1,'ymax':5 }
                ]
SUMMER = [ {'xmin':-39,'xmax':-1,'ymax':7,'ymin':4.5},
            {'xmin':-39,'xmax':-1,'ymax':4.5,'ymin':1},
            {'xmin':-39,'xmax':-1,'ymax':1,'ymin':-2},
            {'xmin':-2,'xmax':2,'ymax':6.5,'ymin':-1},
            {'xmin':-45,'xmax':-38,'ymax':6.5,'ymin':-1}]

def summer_align(xy):
    import math
    xy = xy[:,0:2].copy().transpose() # Grid
    myx = np.mean(xy,axis=1).reshape(2,1)

    #print(xy)
    xyy= xy - myx
    theta = math.radians(-4) # Align the map with the grid 

    rot_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                          [ math.sin(theta),  math.cos(theta)]])

    new_xx = rot_matrix.dot(xyy) + myx

    return(new_xx.transpose())

PREPROCESSING = Tr.Compose([Tr.ToTensor()])


def subsampler(universe,num_sub_samples):
        assert  len(universe)>num_sub_samples
        return np.random.randint(0,len(universe),size=num_sub_samples)



def gen_ground_truth(   poses,
                        sequence,
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

    if sequence=='summer':
        poses = summer_align(poses)
        bbox = SUMMER
    elif sequence=='autumn':
        bbox = AUTUMN

    for i in ROI:
    
        _map_   = poses[:i,:]
        pose    = poses[i,:].reshape((1,-1))
        map_frame_idx  = indices[:i]
        dist_meter  = np.sqrt(np.sum((pose -_map_)**2,axis=1))

        dist = dist_meter/ np.max(dist_meter)

        pos_idx = np.where(dist_meter[:i-roi] < pos_range)[0]
        
        if len(pos_idx)>0:
    
            n_coord=  poses[i].shape[0]
            pa = poses[i].reshape(-1,n_coord)
            pp = poses[pos_idx].reshape(-1,n_coord)
            
            an_labels, an_point_idx = get_roi_points(pa,bbox)
            pos_labels, pos_point_idx = get_roi_points(pp,bbox)

            boolean_sg = np.where(an_labels[0] == pos_labels)[0]
            if len(boolean_sg):
                pos = [pos_idx[pos_point_idx[idx]] for idx in boolean_sg][0]
                min_sort = np.argsort(dist_meter[pos])
                positive.append(pos[min_sort])
                anchor.append(i)
    
    # Negatives
    negatives= []
    neg_idx = np.arange(num_neg)   
    for a, pos in zip(anchor,positive):
        pa = poses[a,:].reshape((1,-1))
        dist_meter = np.linalg.norm(pa-poses,axis=1)
        neg_idx = np.where(dist_meter > neg_range)[0]
        neg_idx = np.setxor1d(neg_idx,pos)
        select_neg = np.random.randint(0,len(neg_idx),num_neg)
        neg_idx = neg_idx[select_neg]
        negatives.append(neg_idx)

    return(anchor,positive,negatives)


def get_roi_points(points,rois):
    #points = points.reshape((-1,2))
    labels = []
    point_idx = []
    for i,roi in enumerate(rois):
        # 
        xmin = (points[:,0]>=roi['xmin'])
        xmax = (points[:,0]<roi['xmax'])
        xx = np.logical_and(xmin, xmax)

        ymin = (points[:,1]>=roi['ymin']) 
        ymax = (points[:,1]<roi['ymax'])
        yy = np.logical_and(ymin, ymax)
        
        selected = np.logical_and(xx, yy)
        idx = np.where(selected==True)[0]
        
        if len(idx)>0:
            labels.append(i)
            point_idx.append(idx)
   
    return np.array(labels), np.array(point_idx)


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
        self.num_pos = ground_truth['num_pos']
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
       
        
        self.anchors,self.positives,self.negatives = gen_ground_truth(self.pose,seq,**ground_truth)
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
        if sequence == 'autumn':
            self.line_rois = AUTUMN
        else:
            self.line_rois = SUMMER

        self.num_samples = self.point_cloud_files.shape[0]
        self.idx_universe = np.arange(self.num_samples)

        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)

        self.poses = self._get_pose_()
        #assert len(np.intersect1d(self.anchors,self.map_idx)) == 0, 'No indicies should be in both anchors and map'
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

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_modality_(i)
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
                        **argv):

        super(ORCHARDSTriplet,self).__init__(root,dataset,sequence, sync = sync, modality=modality,**argv)

        self.modality = modality
        self.aug_flag = aug
        self.mode = mode
        self.eval_mode = False

        self.preprocessing = PREPROCESSING
        verbose = True
    

        # Triplet data
        self.num_samples = len(self._get_point_cloud_file_())
        self.idx_universe = np.arange(self.num_samples)
        # Eval data
        self.poses = self._get_pose_()

         # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    
    def load_subset(self,subsetidx):
        
        self.anchors= np.array(self.anchors)[subsetidx]
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
               
        
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
        pos_idx = pos_idx[:self.num_pos]
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
                                            )

        if split_mode == 'cross-val':
            # Cross-validation. Train and test sets are from different sequences
            test_set = ORCHARDSEval( root =  kwargs['root'],
                                        mode = kwargs['mode'],
                                        **test_loader['data'],
                                        ground_truth = test_loader['ground_truth']
                                        )
         

        elif  split_mode == 'train-test':
            # train-test: train and test sets are from the same sequences, which is split randomly in two.
            # Before
            train_size = 0.6
            print('Train data set:', len(train_set))
            print("train-test split: " + str(train_size))
            # Random split
            train_set_size = int(len(train_set) * train_size)
            valid_set_size = len(train_set) - train_set_size
            train_set, test_set = data.random_split(train_set, [train_set_size, valid_set_size])
            
            test_set.dataset = copy.copy(test_set.dataset) # Mandatory to guarantee independence from the training
            train_set.dataset.load_subset(train_set.indices)
            
            print("Train set: " + str(len(train_set.indices)))
            print("Test set: " + str(len(test_set.indices)))
            train_set = train_set.dataset

            test_set.dataset.set_eval_mode(True)
            test_set.dataset.load_subset(test_set.indices)
            test_set = test_set.dataset
            
        else:
            # Test on the same dataset as trained (Only for debugging)
            test_set =  copy.copy(train_set)
            test_set.set_eval_mode(True)

    
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
    

def conv2PIL(image):
    min_val = np.min(image)
    max_val = np.max(image)
    value = (image-min_val)
    nomr_val = (value/np.max(value))*255
    im_pil = Image.fromarray(nomr_val)
    im_pil = im_pil.convert("L")
    return(im_pil)

   


