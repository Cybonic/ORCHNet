
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
import pandas as pd
import pickle
import random

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


def load_pc_file(file):
	#returns Nx3 matrix
	pc=np.fromfile(file, dtype=np.float64)

	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_pose_file(filename):
    if not 'locations' in filename:
        filename = fixe_name(filename)

    with open(filename, 'rb') as handle:
        queries = pd.read_csv(handle).to_numpy()[:,1:]
        print("pose Loaded")
    return queries

def fixe_name(filename):
    file_structure = filename.split(os.sep)
    file_path = os.sep.join(file_structure[:-2])
    file_name = file_structure[-2].split('_')

    new_file_name = os.path.join(file_path,file_name[0] + '_' + 'locations' +'_'+ file_name[1] + '_' + file_name[2] + '.csv')
    return(new_file_name)



def load_pc_files(filenames):
    pcs=[]
    if not isinstance(filenames,list):
        filenames = [filenames]

    for filename in filenames:
		#print(filename)
        pc=load_pc_file(filename)
        if(pc.shape[0]!=4096):
            continue
        pcs.append(pc)
    pcs=np.array(pcs)
    return pcs


def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries


def get_query_tuple(root,dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
	#get query tuple for dictionary entry
	#return list [query,positives,negatives]
    query_file  = os.path.join(root,dict_value["query"])
    query = load_pc_files(query_file) #Nx3
    query_pose = dict_value['pose']
    #query = {'pcl':query_pcl,'pose':query_pose}
    random.shuffle(dict_value["positives"])
    
    # ==========================================================================
    # Get Positive files
    pos_files=[]
    pos_poses=[]
    for i in range(num_pos):
        indice = dict_value["positives"][i]
        pos_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
        pos_poses.append(QUERY_DICT[indice]["pose"])
    
    positives  = load_pc_files(pos_files)

    # ==========================================================================
    # Get Negatives
    neg_files=[]
    neg_indices=[]
    neg_poses = []
    if(len(hard_neg)==0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            indice = dict_value["negatives"][i]
            neg_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
            neg_poses.append(QUERY_DICT[indice]["pose"])

            ne = dict_value["negatives"][i]
            neg_indices.append(ne)
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            
            neg_indices.append(i)
        j=0
        while(len(neg_files)<num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                indice = dict_value["negatives"][j]
                neg_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
                neg_poses.append(QUERY_DICT[indice]["pose"])
                neg_indices.append(dict_value["negatives"][j])
                j+=1

    negatives=load_pc_files(neg_files)

    # ==========================================================================
    # Get HARD Positives

    if(other_neg==False):
        return [query,positives,negatives]
	#For Quadruplet Loss
    else:
		#get neighbors of negatives and query
        neighbors=[]
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        
        if(len(possible_negs)==0):
            return [query, positives, negatives, np.array([])]
        
        indice = possible_negs[0]
        neg2= load_pc_files(os.path.join(root,QUERY_DICT[indice]["query"]))
        neg2_poses = QUERY_DICT[indice]["pose"]

    # Original implementation does not return Pose
    return {'pcl':[query,positives,negatives,neg2],'pose':[query_pose,pos_poses,neg_poses,neg2_poses]}



# ===================================================================================================================
#       
#
#
# ===================================================================================================================
def load_picklet(root,filename):
        
    pickle_file = os.path.join(root,filename)
    assert os.path.isfile(pickle_file),'target file does nor exist: ' + pickle_file

    queries = get_queries_dict(pickle_file)
    return queries 


class PointNetDataset():
    def __init__(self,
                    root,
                    pickle_file, # choose between train and test files
                    num_neg   = 10, # num of negative samples
                    num_pos   =  1, # num of positive samples
                    modality  = 'range',
                    image_proj=True,
                    aug = False,
                    **argv):
        
        self.plc_files  = []
        self.plc_names  = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []
        self.modality = modality
        self.aug = aug
        self.num_neg = num_neg
        self.num_pos = num_pos
        #self.ground_truth_mode = argv['ground_truth']

        # Stuff related to the data organization
        self.base_path = os.path.join(root,'benchmark_datasets')
        self.queries = load_picklet(root,pickle_file)
      
        self.num_samples  = len(self.queries.keys())

        # Stuff related to sensor parameters for obtaining final representation
        cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        

        dataset_param = sensor_cfg['oxford']
        sensor =  sensor_cfg[dataset_param['sensor']]

        if modality in ['range','projection','remissions']:
            proj_pram = dataset_param['RP']
            self.proj = SphericalRangeProjScan(**sensor,**proj_pram,roi = dataset_param['roi'],parser = None,**argv)
        elif modality in ['intensity','density','height','bev']:
            proj_pram = dataset_param['BEV']
            self.proj = BirdsEyeViewScan(**proj_pram, roi = dataset_param['roi'], parser = None,image_proj=image_proj,**argv)
    
    def __len__(self):
        return(self.num_samples)

    def _get_proj_(self,idx,modality=None,yaw=None):
        # Get point cloud file
        query = self.queries[idx]
        # The function returns a dict. with the following data 
        # {'pcl':[],'pose':[]}
        # both quies have the same data structure [query,pos,neg,neg2]    
        tuple = get_query_tuple(self.base_path,query,self.num_pos,self.num_neg, self.queries, hard_neg=[], other_neg=True)
        return tuple['pcl'], tuple['pose']

    def _get_pose(self):
        return np.array(self.poses)
    
    def _get_anchor(self):
        return np.array(self.anchors)


class KITTIEval(PointNetDataset):
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
        img,pose = self._get_proj_(idx,self.modality)
        return img,pose

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


class KITTITriplet(PointNetDataset):
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