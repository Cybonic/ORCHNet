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
from utils.retrieval import gen_ground_truth


ASEGMENTS = [   {'xmin':-15,'xmax':-9,'ymin':-50,'ymax':-1 },
                {'xmin':-9,'xmax':-5,'ymin':-50,'ymax':-1 },
                {'xmin':-5,'xmax':-2,'ymin':-50,'ymax':-1 },
                {'xmin':-2,'xmax':2,'ymin':-50,'ymax':-1 },
                {'xmin':-15,'xmax':2,'ymin':-55,'ymax':-49 },
                {'xmin':-15,'xmax':2,'ymin':-3,'ymax':5 }
                ]



                
PREPROCESSING = Tr.Compose([Tr.ToTensor()])

MODALITIES = ['range','projection','remissions','mask']


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


def get_roi_points(points,roi):
    roi_dx = ((points[:,0]>=roi['xmin']).astype(np.int8) * (points[:,0]<roi['xmax']).astype(np.int8) *
                  (points[:,1]>=roi['ymin']).astype(np.int8) * (points[:,1]<roi['ymax']).astype(np.int8))
    
    return roi_dx.astype(np.bool8)



def comp_line_gt_table(pose,rois):
    '''
    returns a array with same size of poses. with labels of each line
    '''
    segments = np.zeros(pose.shape[0],dtype=np.uint8)
    for i, roi in enumerate(rois):
        roi_dx = get_roi_points(pose,roi)
        print(roi_dx.sum())
        segments[roi_dx] = i 
    return(segments)




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
    def __init__(self,root,dataset,seq,sync = True , modality='pcl' ,**argv):
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
        
        # Load indicies to split the dataset in queries, positive and map 
        triplet_file = os.path.join(self.target_dir,'sync_triplets.txt')
        assert os.path.isfile(triplet_file), 'Triplet indice file does not exist: ' + triplet_file
        self.anchors, _ , _ = parse_triplet_file(triplet_file)

    
    def __len__(self):
        return self.pose.shape[0]

    def __call__(self,idx):
        return self.load_point_cloud(idx)

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
                num_subsamples = -1,
                mode = 'Disk', 
                **argv
                ):
        
        super(ORCHARDSEval,self).__init__(root, dataset, sequence, sync=sync, modality=modality,**argv)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING

        self.num_samples = self.point_cloud_files.shape[0]
        # generate map indicies:ie inidices that do not belong to the anchors/queries
        self.idx_universe = np.arange(self.num_samples)

        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
        self.poses = self._get_pose_()
        self.gt_table = comp_gt_table(self.poses,self.anchors,argv['pos_range'])
        self.line_gt_table = comp_line_gt_table(self.pose,ASEGMENTS)

        if num_subsamples > 0:
            self.set_subsamples(num_subsamples)
        
        assert len(np.intersect1d(self.anchors,self.map_idx)) == 0, 'No indicies should be in both anchors and map'

        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    def compt_gt_table(self,anchor,positive,num_samples):
        gt_mapping = np.zeros((len(anchor),num_samples),np.uint8)
        for i,p in enumerate(positive):
            gt_mapping[i,p]=1
        return(gt_mapping)

    def get_GT_Map(self):
        return(self.gt_table)
    
    def _get_representation_(self,idx):
        data,_ = self._get_modality_(idx,self.modality)
        return data

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_representation_(i)
            img[i]=data#.astype(np.uint8)
        return img

    def get_data(self,index):
        global_index = self.idx_universe[index] # Only useful when subsampler is on
        
        if self.mode == 'RAM':
            data = self.inputs[global_index]         
        elif self.mode == 'Disk':
            data = self._get_representation_(global_index)
        
        plc = self.preprocessing(data)
        return(plc,global_index)

    def __getitem__(self,index):
        pcl,gindex = self.get_data(index)
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
    





class ORCHARDSTriplet(OrchardDataset):
    def __init__(self,
                        root,
                        dataset,
                        sequence, 
                        sync = True, 
                        mode='Disk', 
                        modality = 'projection', 
                        aug=False,
                        pos_thres = 2, # Postive range threshold; positive samples  < pos_thres
                        neg_thres = 5, # Range threshold  for negative samples; negative samples > neg_thres 
                        num_neg = 20 , # Number of negative samples to fetch
                        num_pos = 1,  # Number of positive samples to fetch
                        num_subsamples = 0,
                        **argv):

        super(ORCHARDSTriplet,self).__init__(root,dataset,sequence, sync = sync, modality=modality,**argv)

        self.modality = modality
        self.aug_flag = aug
        self.mode = mode
        self.eval_mode = False

        self.preprocessing = PREPROCESSING
        
        # Triplet data
        self.num_samples = len(self._get_point_cloud_file_())
        self.idx_universe = np.arange(self.num_samples)
        self.positive , self.negative = gen_ground_truth(self.pose,self.anchors,pos_thres,neg_thres,num_neg,num_pos)

        # Eval data
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
        self.poses = self._get_pose_()
        self.gt_table = comp_gt_table(self.poses,self.anchors,pos_thres)
        self.line_gt_table = comp_line_gt_table(self.pose,ASEGMENTS)

        # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()
        
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
        an_idx,pos_idx,neg_idx  = self.anchors[index],self.positive[index], self.negative[index]
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
        return(len(self.anchors))
    
    def get_pose(self):
        return self._get_pose_()

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)
    
    def get_GT_Map(self):
        return(self.gt_table)




# ===================================================================================================================
#
#
#
# ===================================================================================================================


class ORCHARDS():
    def __init__(self,train_loader,test_loader, split_mode='cross-val', **kwargs):

        self.valloader = None
        self.trainloader = None

        assert split_mode in ['cross-val','train-test'], "Split mode not recognized: " + split_mode
        import copy

        train_set = ORCHARDSTriplet(root = kwargs['root'],
                                            mode = kwargs['mode'],
                                            **train_loader['data']
                                            )

        if split_mode == 'cross-val':
            # Cross-validation. Train and test sets are from different sequences
            if 'val_loader' in kwargs:
               
                test_set = ORCHARDSEval( root =  kwargs['root'],
                                                mode = kwargs['mode'],
                                                #num_subsamples = num_subsamples,
                                                **test_loader['data'])

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
            test_set.dataset =  copy.copy(train_set.dataset)
            test_set.dataset.set_eval_mode(True)
        
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

   


