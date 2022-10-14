
import os
from tqdm import tqdm
import numpy as np
from utils.retrieval import gen_ground_truth, comp_gt_table
import yaml
from .sphericalscan import SphericalRangeProjScan
from .birdsviewscan import BirdsEyeViewScan
from torch.utils.data import DataLoader
import torchvision.transforms as Tr
import torch
from .laserscan import LaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def subsampler(universe,num_sub_samples):
    if not  len(universe)>num_sub_samples:
        num_sub_samples = len(universe)
    return np.random.randint(0,len(universe),size=num_sub_samples)

def parse_triplet_file(file):
    assert os.path.isfile(file)
    f = open(file)
    anchors = []
    positives = []
    negatives = []
    for line in f:
        value_str = line.rstrip().split('_')
        anchors.append(int(value_str[0].split(':')[-1]))
        positives.append(int(value_str[1].split(':')[-1]))
        negatives.append([int(i) for i in value_str[2].split(':')[-1].split(' ')])
    f.close()

    anchors = np.array(anchors,dtype=np.uint32)
    positives = np.array(positives,dtype=np.uint32)
    negatives = np.array(negatives,dtype=np.uint32)

    return anchors,positives,negatives

def load_indices(file):
    overlap = []
    for f in open(file):
        f = f.split(':')[-1]
        indices = [int(i) for i in f.split(' ')]
        overlap.append(indices)
    return(overlap[0])


def load_pose_to_RAM(file):
    assert os.path.isfile(file)
    pose_array = []
    for line in tqdm(open(file), 'Loading to RAM'):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        position = values[[3,7,11]]
        #position[:,1:] =position[:,[2,1]] 
        pose_array.append(position.tolist())

    pose_array = np.array(pose_array)   
    pose_array[:,1:] =pose_array[:,[2,1]] 
    return(pose_array)


def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([os.path.join(target_dir.split(os.sep)[-2],f.split('.')[0]) for f in os.listdir(target_dir)])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])
    return(files[idx],fullfiles[idx])



class kitti_velo_parser():
    def __init__(self):
        self.dt = []

    def velo_read(self,scan_path):
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return(np.array(scan))

# ===================================================================================================================
#       
#
#
# ===================================================================================================================
class FileStruct():
    def __init__(self,root,dataset,sequence,sync = True):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        #for seq in sequences:
        self.target_dir = os.path.join(root,dataset,sequence)
        #self.target_dir.append(target_dir)
        assert os.path.isdir(self.target_dir),'target dataset does nor exist: ' + self.target_dir

        pose_file = os.path.join(self.target_dir,'poses.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_pose_to_RAM(pose_file)
        #self.pose.extend(pose)

        point_cloud_dir = os.path.join(self.target_dir,'velodyne')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        self.file_names, self.point_cloud_files = get_files(point_cloud_dir)

    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files,self.file_names)
        return(self.point_cloud_files[idx],self.file_names[idx])
    
    def _get_pose_(self,idx=np.nan):
        if not isinstance(idx,(np.ndarray, np.generic)):
            idx = np.array(idx)
        if idx.size==1 and np.isnan(idx):
            return(self.pose)
        return(self.pose[idx])
    
    def _get_target_dir(self):
        return(self.target_dir)


class KittiDataset():
    def __init__(self,
                        root,
                        dataset,
                        sequence,
                        modality,
                        max_points=50000, 
                        pos_range = 10, # max positive range
                        neg_range = 50, # min negative range
                        num_neg   = 10, # num of negative samples
                        num_pos   = 20, # num of positive samples
                        image_proj=True,
                        aug = False,
                        **argv):
        
        self.plc_files  = []
        self.plc_names  = []
        self.poses      = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []
        self.modality = modality
        baseline_idx  = 0 
        self.max_points = max_points
        self.aug = aug
        self.ground_truth_mode = argv['ground_truth']
        assert isinstance(sequence,list)

        for seq in sequence:
            kitti_struct = FileStruct(root, dataset, seq)
            files,name = kitti_struct._get_point_cloud_file_()
            self.plc_files.extend(files)
            self.plc_names.extend(name)
            pose = kitti_struct._get_pose_()
            self.poses.extend(pose)
            target_dir = kitti_struct._get_target_dir()
            # Load indicies to split the dataset in queries, positive and map 
            triplet_file = os.path.join(target_dir,'sync_triplets.txt')
            
            assert os.path.isfile(triplet_file), 'Triplet indice file does not exist: ' + triplet_file
            anchor, _ , _ = parse_triplet_file(triplet_file)
            positive , negative = gen_ground_truth(pose,anchor,pos_range,neg_range,num_neg,num_pos,mode=self.ground_truth_mode)
            
            self.anchors.extend(baseline_idx + anchor)
            self.positives.extend(baseline_idx + positive)
            self.negatives.extend(baseline_idx + negative)

            baseline_idx += len(files)
        # Load dataset and laser settings
        self.poses = np.array(self.poses)
        cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        
        self.num_samples = len(self.plc_files)
        dataset_param = sensor_cfg[seq]
        sensor =  sensor_cfg[dataset_param['sensor']]

        if modality in ['range','projection','remissions']:
            proj_pram = dataset_param['RP']
            self.proj = SphericalRangeProjScan(**sensor,**proj_pram,roi = dataset_param['roi'],parser = kitti_velo_parser(),**argv)
        elif modality in ['intensity','density','height','bev']:
            proj_pram = dataset_param['BEV']
            self.proj = BirdsEyeViewScan(**proj_pram, roi = dataset_param['roi'], parser = kitti_velo_parser(),image_proj=image_proj,**argv)
        else:
            self.proj = LaserScan(parser = kitti_velo_parser(), max_points = self.max_points, **argv)
    
    def __len__(self):
        return(self.num_samples)

    def _get_proj_(self,idx,modality=None,yaw=None):
        # Get point cloud file
        file = self.plc_files[idx]
        self.proj.open_scan(file)
        return self.proj.get_data(modality = self.modality, aug = self.aug),None

    def _get_pose(self):
        return np.array(self.poses)
    
    def _get_anchor(self):
        return np.array(self.anchors)


class KITTIEval(KittiDataset):
    def __init__(self,root, dataset, sequence,   # Projection param and sensor
                modality = 'range' , 
                num_subsamples = -1,
                mode = 'Disk',
                max_points = 10000, 
                image_proj=True,
                **arg
                ):
        
        super(KITTIEval,self).__init__(root, dataset, sequence, modality=modality,max_points=max_points,**arg)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING

        self.num_samples = self.num_samples
        # generate map indicies:ie inidices that do not belong to the anchors/queries
        self.idx_universe = np.arange(self.num_samples)

        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors) 
        # Build ground truth table  
        #self.positives , _ = gen_ground_truth(self.poses,self.anchors,5,0,0,12) # pos_thres,neg_thres,num_neg,num_pos
        self.poses = self._get_pose()
        self.gt_table = comp_gt_table(self.poses,self.anchors,arg['pos_range'])
         # Selection of a smaller sample size for debugging
        if num_subsamples > 0:
            self.set_subsamples(num_subsamples)

        assert len(np.intersect1d(self.anchors,self.map_idx)) == 0, 'No indicies should be in both anchors and map'
        
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    def set_subsamples(self,samples):
        anchor_subset_idx = subsampler(self.anchors,samples)
        self.anchors  = np.sort(np.array(self.anchors,dtype=np.uint32)[anchor_subset_idx])
        subset_idx    = subsampler(self.map_idx,samples)
        self.map_idx  = np.sort(np.array(self.map_idx,dtype=np.uint32)[subset_idx])
        #self.gt_table = self.gt_table[anchor_subset_idx,:] # Get only the gt loop from the subset
        self.idx_universe   = np.sort(np.concatenate((self.anchors,self.map_idx)))
        self.poses = np.array(self.poses)[self.idx_universe]
        self.num_samples = len(self.idx_universe)

    def compt_gt_table(self,anchor,positive,num_samples):
        gt_mapping = np.zeros((len(anchor),num_samples),np.uint8)
        for i,p in enumerate(positive):
            gt_mapping[i,p]=1
        return(gt_mapping)

    def get_GT_Map(self):
        return(self.gt_table)
    
    def _get_representation_(self,idx):
        img,_ = self._get_proj_(idx,self.modality)
        return img

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            img[i] = self._get_representation_(i)#.astype(np.uint8)
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
        pcl,index = self.get_data(index)
        
        return(pcl,index.astype(np.float32))

    def __len__(self):
        return(self.num_samples)
        
    def get_pose(self):
        return np.array(self.poses)

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)


class KITTITriplet(KittiDataset):
    def __init__(self,
                root,
                dataset,
                sequence,
                sync = True, 
                mode='Disk', 
                modality = 'projection', 
                aug=False, 
                num_subsamples = 0,
                max_points=10000,
                image_proj=True,
                **argv):
        super(KITTITriplet,self).__init__(root,dataset,sequence, modality=modality,max_points=max_points,image_proj=image_proj,aug=aug,**argv)

        self.modality = modality
        self.aug_flag = aug
        self.mode     = mode
        self.preprocessing = PREPROCESSING
        # Select randomly a sub set
        #self.num_samples = len(self._get_point_cloud_file_())
        self.idx_universe = np.arange(self.num_samples)
        if num_subsamples > 0:
            self.set_subsampler(num_subsamples)
        # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    def set_subsampler(self,samples):
        subset_idx     = subsampler(self.anchors,samples)
        self.anchors   = np.array(self.anchors)[subset_idx]
        self.positives = np.array(self.positives)[subset_idx]
        self.negatives = np.array(self.negatives)[subset_idx]

        from utils.utils  import unique2D
        positive = unique2D(self.positives) 
        negative = unique2D(self.negatives)
        self.idx_universe =np.unique(np.concatenate((self.anchors,positive,negative)))
        self.num_samples = len(self.idx_universe)

    def load_RAM(self):
        img   = {}        
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data= self._get_representation_(i)#.astype(np.uint8)
            img[i]=data#.astype(np.uint8)
        return img

    def get_data(self,index):
        an_idx,pos_idx,neg_idx = self.anchors[index],self.positives[index], self.negatives[index]
        if self.mode == 'RAM':     
            # point clouds are already converted to the input representation, is only required to 
            #  convert to tensor 
            plt_anchor = self.preprocessing(self.inputs[an_idx])
            plt_pos = torch.stack([self.preprocessing(self.inputs[i]) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self.inputs[i]) for i in neg_idx],axis=0)

        elif self.mode == 'Disk':
            plt_anchor = self.preprocessing(self._get_representation_(an_idx))
            plt_pos = torch.stack([self.preprocessing(self._get_representation_(i)) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self._get_representation_(i)) for i in neg_idx],axis=0)
          
        else:
            raise NameError

        an_poses  = self.poses[an_idx].reshape(-1,3).astype(np.float32)
        pos_poses = self.poses[pos_idx].reshape(-1,3).astype(np.float32)
        neg_poses = self.poses[neg_idx].reshape(-1,3).astype(np.float32)

        pcl  = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        #indx = {'anchor':an_idx.astype(np.int32),'positive':pos_idx.astype(np.int32),'negative':neg_idx.astype(np.int32)}
        indx = {'anchor':an_poses,'positive':pos_poses,'negative':neg_poses}
        #indx = {'anchor':len(plt_anchor),'positive':len(plt_pos),'negative':len(plt_neg)}
        return(pcl,indx)

    def _get_representation_(self,idx):
        img,_ = self._get_proj_(idx,self.modality)
        return img

    def __getitem__(self,index):
        pcl,indx = self.get_data(index)
        #indx =torch.tensor(indx)
        return(pcl,indx)

    def get_pose(self):
        return np.array(self.poses)

    def __len__(self):
        return(len(self.anchors))


class KITTI():
    def __init__(self,**kwargs):

        self.valloader   = None
        self.trainloader = None
        num_subsamples = -1
        debug = False

        if 'debug' in kwargs and kwargs['debug'] == True:
            num_subsamples = 100
            print("[Kitti] Debug mode On Training and Val data are the same")
            debug  =True 

        if 'num_subsamples' in kwargs:
            num_subsamples = kwargs['num_subsamples']

        if 'val_loader' in kwargs:
            val_cfg   = kwargs['val_loader']
            

            self.val_loader = KITTIEval( root =  kwargs['root'],
                                            mode = kwargs['mode'],
                                            num_subsamples = num_subsamples,
                                            **val_cfg['data']
                                            )

            self.valloader   = DataLoader(self.val_loader,
                                    batch_size = val_cfg['batch_size'],
                                    num_workers= 0,
                                    pin_memory=True,
                                    )

        if 'train_loader'  in kwargs:

            train_cfg = kwargs['train_loader']

            self.train_loader = KITTITriplet(root =  kwargs['root'],
                                                mode = kwargs['mode'], 
                                                num_subsamples = num_subsamples,
                                                **train_cfg['data'],
                                                )

            self.trainloader   = DataLoader(self.train_loader,
                                        batch_size = 1, #train_cfg['batch_size'],
                                        shuffle    = train_cfg['shuffle'],
                                        num_workers= 0,
                                        pin_memory=True,
                                        drop_last=True,
                                        )
        
        # For debugging Use the same data 
        #if debug:
        #    self.valloader = self.trainloader


    def get_train_loader(self):
        return self.trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    def get_val_loader(self):
        return  self.valloader
    
    def get_label_distro(self):
        raise NotImplemented
		#return  1-np.array(self.label_disto)