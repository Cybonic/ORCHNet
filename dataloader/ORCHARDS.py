import os

from PIL import Image 
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as Tr
from  torch.utils.data.sampler import SubsetRandomSampler
import yaml
import torch
#from utils.retrieval import euclidean_knn 

from .sphericalscan import SphericalRangeProjScan
from .birdsviewscan import BirdsEyeViewScan
from .laserscan import LaserScan

from utils.retrieval import gen_ground_truth

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

MODALITIES = ['range','projection','remissions','mask']

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
        values = [float(v) for v in values_str]
        pose_array.append(values[0:3])
    return(np.array(pose_array))

def load_to_RAM(file):
    assert os.path.isfile(file)
    pose_array = []
    for line in tqdm(open(file), 'Loading to RAM'):
        values_str = line.split(' ')
        values = [float(v) for v in values_str]
        pose_array.append(values[0:3])
    return(np.array(pose_array))

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

def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([f.split('.')[0] for f in os.listdir(target_dir)])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])
    return(files[idx],fullfiles[idx])

def get_point_cloud_files(dir):
    path = dir
    if not os.path.isdir(path):
        print("[ERR] Path does not exist!")
    files = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return(files)

def subsampler(universe,num_sub_samples):
        assert  len(universe)>num_sub_samples
        return np.random.randint(0,len(universe),size=num_sub_samples)


# ===================================================================================================================
#       
#
#
# ===================================================================================================================
class parser():
    def __init__(self):
        self.dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2'), ('time', '<f4')]
    
    def velo_read(self,file):
        input_file = open(file,'rb')
        scan = np.fromfile(input_file, dtype=self.dt)
        scan = np.hstack((scan['x'].reshape((-1,1)),scan['y'].reshape((-1,1)),scan['z'].reshape((-1,1)),scan['intensity'].reshape((-1,1))))
        return scan

# ===================================================================================================================
#       
#
#
# ===================================================================================================================
class FileStruct():
    def __init__(self,root,dataset,sequence,sync = True):
        self.target_dir = os.path.join(root,dataset,sequence)
        assert os.path.isdir(self.target_dir),'target dataset does nor exist: ' + self.target_dir

        pose_file = os.path.join(self.target_dir,'poses.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_pose_to_RAM(pose_file)

        point_cloud_dir = os.path.join(self.target_dir,'point_cloud')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        self.file_name, self.point_cloud_files = get_files(point_cloud_dir)

        
        if sync == True:
            sync_plc_idx_file = os.path.join(self.target_dir,'sync_point_cloud_idx.txt')
            assert os.path.isfile(pose_file),'sync plc file does not exist: ' + sync_plc_idx_file

            sync_pose_idx_file = os.path.join(self.target_dir,'sync_pose_idx.txt')
            assert os.path.isfile(pose_file),'sync pose file does not exist: ' + sync_pose_idx_file

            sync_pose_idx = load_indices(sync_pose_idx_file)
            sync_plc_idx = load_indices(sync_plc_idx_file)

            self.point_cloud_files = self.point_cloud_files[sync_plc_idx]
            self.pose =  self.pose[sync_pose_idx]
    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files,self.file_name)
        return(self.point_cloud_files[idx],self.file_name[idx])
    
    def _get_pose_(self,idx=np.nan):
        if not isinstance(idx,(np.ndarray, np.generic)):
            idx = np.array(idx)
        if idx.size==1 and np.isnan(idx):
            return(self.pose)
        return(self.pose[idx])


# ===================================================================================================================
#       
#
#
# ===================================================================================================================

class OrchardDataset(FileStruct):
    def __init__(self,  root,
                        dataset,
                        sequence, 
                        modality ,
                        max_points = 10000, 
                        sync = True,
                        pos_thres = 1, # Postive range threshold; positive samples  < pos_thres
                        neg_thres = 3, # Range threshold  for negative samples; negative samples > neg_thres 
                        num_neg = 50 , # Number of negative samples to fetch
                        num_pos = 12,  # Number of positive samples to fetch
                        image_proj = True, # Project to an image pixel (0 - 255)
                        **argv
                        ):
        super(OrchardDataset,self).__init__(root, dataset, sequence, sync = sync)
        
        self.modality = modality
        self.max_points = max_points

        # Load indicies to split the dataset in queries, positive and map 
        triplet_file = os.path.join(self.target_dir,'sync_triplets.txt')
        assert os.path.isfile(triplet_file), 'Triplet indice file does not exist: ' + triplet_file
        self.anchor, _ , _ = parse_triplet_file(triplet_file)
        self.positive , self.negative = gen_ground_truth(self.pose,self.anchor,pos_thres,neg_thres,num_neg,num_pos)
        
        # Load dataset and laser settings
        cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        
        dataset_param = sensor_cfg[sequence]
        sensor =  sensor_cfg[dataset_param['sensor']]

        if modality in ['range','projection','remissions']:
            proj_pram = dataset_param['RP']
            self.proj = SphericalRangeProjScan(**sensor,**proj_pram,parser = parser())
        elif modality in ['intensity','density','height','bev']:
            proj_pram = dataset_param['BEV']
            self.proj = BirdsEyeViewScan(**proj_pram, parser = parser(),image_proj = image_proj)
        else:
            self.proj = LaserScan(parser = parser(),max_points=max_points,**argv)

    def _get_pose(self):
        return np.array(self.pose)
    
    def _get_anchor(self):
        return np.array(self.anchor)

    def _get_proj_(self,idx,modality=None,yaw=None):
        #assert modality in MODALITIES
        # Get point cloud file
        file,plc_name = self._get_point_cloud_file_(idx)
        self.proj.open_scan(file)
        return self.proj.get_data(modality = self.modality),plc_name

    def get_pcl(self,idx):
        return self._get_proj_(idx,'points')
        
    def get_projections(self,idx):
        return self._get_proj_(idx,'projection')
    
    def get_range(self,idx):
        return self._get_proj_(idx,'range')
    
    def get_remissions(self,idx):
        return self._get_proj_(idx,'remissions')
    
    def get_mask(self,idx):
        return self._get_proj_(idx,'mask')

    def get_rotated_range_projection(self,idx):
        project= []
        file = self.point_cloud_files[idx]

        self.laser.open_scan(file,90)
        project.append(self.laser.get_data('range'))
        self.laser.open_scan(file,180)
        project.append(self.laser.get_data('range'))
        self.laser.open_scan(file,270)
        project.append(self.laser.get_data('range'))
        project = np.stack(project,axis=-1)
        return project
    
    def __len__(self):
        return(self.point_cloud_files.shape[0])


# ===================================================================================================================
#       
#
#
# ===================================================================================================================


class ORCHARDSEval(OrchardDataset):
    def __init__(self,root, dataset, sequence, sync = True,   # Projection param and sensor
                modality = 'range' , 
                num_subsamples = -1,
                mode = 'Disk', 
                mean = [0,0,0],
                std  = [1,1,1],
                ):
        
        super(ORCHARDSEval,self).__init__(root, dataset, sequence, sync=sync, modality=modality)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING

        self.num_samples = self.point_cloud_files.shape[0]
        # generate map indicies:ie inidices that do not belong to the anchors/queries
        self.idx_universe = np.arange(self.num_samples)

        self.map_idx  = np.setxor1d(self.idx_universe,self.anchor)

        # Build ground truth table
        self.positive , _ = gen_ground_truth(self.pose,self.anchor,1,0,0,12)
        self.gt_table = self.compt_gt_table(self.anchor,self.positive,self.num_samples)
        self.gt_table  =self.gt_table[:,self.map_idx]
         # Selection of a smaller sample size for debugging
        if num_subsamples > 0:
            self.set_subsamples(num_subsamples)
        
        assert len(np.intersect1d(self.anchor,self.map_idx)) == 0, 'No indicies should be in both anchors and map'

        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    def set_subsamples(self,samples):
        anchor_subset_idx = subsampler(self.anchor,samples)
        self.anchor = np.array(self.anchor)[anchor_subset_idx]
        
        subset_idx  = subsampler(self.map_idx,samples)
        self.map_idx = np.array(self.map_idx)[subset_idx]
        
        self.gt_table = self.gt_table[anchor_subset_idx,:]

        self.idx_universe   = np.concatenate((self.anchor,self.map_idx))
        self.num_samples = len(self.idx_universe)
        #print()

    def compt_gt_table(self,anchor,positive,num_samples):
        gt_mapping = np.zeros((len(anchor),num_samples),np.uint8)
        for i,p in enumerate(positive):
            gt_mapping[i,p]=1
        return(gt_mapping)

    def get_GT_Map(self):
        return(self.gt_table)
    
    def _get_representation_(self,idx):
        data,_ = self._get_proj_(idx,self.modality)
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
        return np.array(self.anchor,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)



# ===================================================================================================================
#
#
#
# ===================================================================================================================



class ORCHARDSTriplet(OrchardDataset):
    def __init__(self,root,dataset,sequence, sync = True, mode='Disk', modality = 'projection', aug=False, num_subsamples = 0):
        super(ORCHARDSTriplet,self).__init__(root,dataset,sequence, sync = sync, modality=modality)

        self.modality = modality
        self.aug_flag = aug
        self.mode = mode
        self.preprocessing = PREPROCESSING
        # Select randomly a sub set
        self.num_samples = len(self._get_point_cloud_file_())
        self.idx_universe = np.arange(self.num_samples)
        if num_subsamples > 0:
            self.set_subsampler(num_subsamples)
        # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

    def set_subsampler(self,samples):
        subset_idx = subsampler(self.anchor,samples)
        # Select subsets
        self.anchor =  np.array(self.anchor)[subset_idx]
        self.positive = np.array(self.positive)[subset_idx]
        self.negative = np.array(self.negative)[subset_idx]
        # Filter repeated samples to maintain memory footprint low
        from utils.utils  import unique2D
        positive = unique2D(self.positive) 
        negative = unique2D(self.negative)
        self.idx_universe =np.unique(np.concatenate((self.anchor,positive,negative)))
        self.num_samples = len(self.idx_universe)
        
    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_representation_(i)
            img[i]=data#.astype(np.uint8)
        return img

    def get_data(self,index):
        an_idx,pos_idx,neg_idx  = self.anchor[index],self.positive[index], self.negative[index]
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

        pcl = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':len(plt_anchor),'positive':len(plt_pos),'negative':len(plt_neg)}
        return(pcl,indx)

    def _get_representation_(self,idx):
        data,_ = self._get_proj_(idx,self.modality)
        return data

    def __getitem__(self,index):
        pcl,indx = self.get_data(index)
        return(pcl,indx)

    def __len__(self):
        return(len(self.anchor))




# ===================================================================================================================
#
#
#
# ===================================================================================================================


class ORCHARDS():
    def __init__(self,**kwargs):

        self.valloader = None
        self.trainloader = None
        num_subsamples = -1
        
        if 'debug' in kwargs and kwargs['debug'] == True:
            num_subsamples = 50
        if 'num_subsamples' in kwargs:
            num_subsamples = kwargs['num_subsamples']

        if 'val_loader' in kwargs:
            val_cfg   = kwargs['val_loader']
            

            self.val_loader = ORCHARDSEval( root =  kwargs['root'],
                                            mode = kwargs['mode'],
                                            #num_subsamples = num_subsamples,
                                            **val_cfg['data'])

            self.valloader   = DataLoader(self.val_loader,
                                    batch_size = val_cfg['batch_size'],
                                    num_workers= 0,
                                    pin_memory=True,
                                    )

        if 'train_loader'  in kwargs:

            train_cfg = kwargs['train_loader']

            self.train_loader = ORCHARDSTriplet(root =  kwargs['root'],
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
            


    def get_train_loader(self):
        return self.trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    def get_val_loader(self):
        return  self.valloader
    
    def get_label_distro(self):
        raise NotImplemented
		#return  1-np.array(self.label_disto)
    
def conv2PIL(image):
    min_val = np.min(image)
    max_val = np.max(image)
    value = (image-min_val)
    nomr_val = (value/np.max(value))*255
    im_pil = Image.fromarray(nomr_val)
    im_pil = im_pil.convert("L")
    return(im_pil)

if __name__ == "__main__":

    root = '/home/tiago/Dropbox/research/datasets'
    dataset = 'orchard-uk'
    sequence = 'rerecord_sparce'
    modality = {'range':0,'intensity':4}
    
    cfg_file = os.path.join('cfg','overlap_cfg.yaml')
    
    import yaml 
    from utils.laserscanvis import LaserScanVis
    cfg = yaml.load(open(cfg_file),Loader=yaml.loader.SafeLoader)[sequence]
    dataset=ORCHARDSTriplet(root,dataset,sequence,sensor = cfg['sensor'])
    viz = LaserScanVis(dataset=dataset,size=3)
    viz.run()
    
    
    for i in range(10):
        pcl, pose  =  dataset[i]
        image = pcl['anchor'][:,:,modality['intensity']]
        im_pil = conv2PIL(image)
    #im_pil.show()
        im_pil.save(f'temp/{i}.jpg')
   


