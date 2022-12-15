
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


def gather_files(queries:list):
    file_buffer = []
    for k,v in queries.items():
        file_buffer.append(v['query'])
    
    return file_buffer


def get_query_tuple(root,dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    """
        This function returns two a dict. with the following data 
        # {'pcl':[],'pose':[]}
        # both fields have the same data structure [query,pos,neg,neg2] 

    """
	#get query tuple for dictionary entry
	#return list [query,positives,negatives]
    query_file  = os.path.join(root,dict_value["query"])
    query = load_pc_files(query_file) #Nx3
    query_pose = dict_value['pose']

    # ==========================================================================
    # Get Positive files
    random.shuffle(dict_value["positives"])

    pos_files=[]
    pos_poses=[]
    
    if num_pos > len(dict_value["positives"]):
        num_pos = len(dict_value["positives"])

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

    if num_neg > len(dict_value["negatives"]):
        num_neg = len(dict_value["negatives"])

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

    negatives = load_pc_files(neg_files)

    # ==========================================================================
    # Get HARD Negatives

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
    #return {'pcl':[query,positives,negatives,neg2],'pose':[query_pose,pos_poses,neg_poses,neg2_poses]}
    return {'q':query,'p':positives,'n':negatives,'hn':neg2},{'q':query_pose,'p':pos_poses,'n':neg_poses,'hm':neg2_poses}



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
                    num_neg, # num of negative samples
                    num_pos, # num of positive samples
                    num_other,
                    modality,
                    image_proj,
                    aug,
                    max_points,
                    mode='RAM', # mode of loading data: [Disk, RAM]
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
        self.other_neg = num_other
        self.hard_neg = 0
        self.mode = mode # mode of loading data: [Disk, RAM]
        #self.ground_truth_mode = argv['ground_truth']

        self.root = root
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
        else:
            self.proj = LaserScan(parser = None, max_points = max_points, **argv)

        self.file_buffer = gather_files(self.queries)

        if self.mode == 'RAM':
            self.RAM_data = self.load_to_RAM()



    def load_to_RAM(self):
        #file_buffer = gather_files(self.queries)
        self.anchor_idx_buffer = []
        proj_vec = []
        pose_vec = []
        num_samples = len(self.queries.keys()) 
        for i in tqdm(range(num_samples),"Loading to RAM"):
        #for i,file in enumerate(file_buffer):
            
            anchor = self.queries[i]
            if len(anchor['positives'])> 0: # No loop
                self.anchor_idx_buffer.append(i)

            query_file  = os.path.join(self.base_path,anchor['query']) # build the pcl file path
            pcl = load_pc_files(query_file).squeeze()
            self.proj.load_pcl(pcl) # Load the pcl and send to the projection lib
            data = self.proj.get_data(modality = self.modality, aug = self.aug) # map to the representation
            proj_vec.append(data)
            pose_vec.append(anchor['pose'])

        self.num_samples = len(self.anchor_idx_buffer)
        return{'proj':np.array(proj_vec),'pose':np.array(pose_vec)}

    
    def _load_pose_pcl_pair_(self,idx):
        file = self.queries[idx]['query']
        query_file  = os.path.join(self.base_path,file) # build the pcl file path
        pcl = load_pc_files(query_file).squeeze()
        self.proj.load_pcl(pcl) # Load the pcl and send to the projection lib
        proj = self.proj.get_data(modality = self.modality, aug = self.aug) # map to the representation
        pose = self.queries[idx]['pose']

        return {'proj':proj,'pose':pose}

    def __len__(self):
        return(self.num_samples)


    def get_triplet_tuple_idx(self, query_idx: int)-> dict :

        num_pos = self.num_pos 
        num_neg = self.num_neg
        # hard_neg = self.hard_neg
        # other_neg = self.other_neg

        tuple_idx = self.queries[query_idx]
        
        # ------------------------------------------------------
        # Positives samples

        positives = tuple_idx['positives']
        random.shuffle(positives) # Shuffle the positive indices
        # set number of positives to retrieve equal the actual size of positives, 
        # when not enough positives exist
        if num_pos > len(positives):
            num_pos = len(positives)

        # Generate the positive indices
        idx = np.arange(0,self.num_pos,dtype=np.int32)
        pos_idx_vec = np.array(positives)[idx] 

        # ------------------------------------------------------
        # Negative samples
        negatives = tuple_idx['negatives']
        random.shuffle(negatives) # Shuffle the negative indices
        # set number of negatives to retrieve equal the actual size of negatives, 
        # when not enough negatives exist
        if num_neg > len(negatives):
            num_neg = len(negatives)
        
        idx = np.arange(0,self.num_neg,dtype=np.int32)
        neg_idx_vec = np.array(negatives)[idx]

        return {'pos':pos_idx_vec,'neg':neg_idx_vec}	




class PointNetTriplet(PointNetDataset):
    def __init__(self,
                root,
                pickle_file, # choose between train and test files
                num_neg   = 18, # num of negative samples
                num_pos   = 2, # num of positive samples
                other_neg = 1,
                modality  = 'range',
                image_proj= True,
                aug = False,
                num_subsamples = -1,
                mode = 'RAM',
                max_points = 10000, 
                **argv
                ):

        super(PointNetTriplet,self).__init__(root, 
                                            pickle_file, 
                                            num_neg, 
                                            num_pos,
                                            other_neg,
                                            modality, 
                                            image_proj, 
                                            aug, 
                                            max_points,
                                            mode=mode,
                                            **argv)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING

        
    def get_tuple_data(self,idx: int):
        """
        Given a indice, this function returns the respective pose and representation

        args: 
            idx: (int) indice of the data sample
        
        return: 
            proj: (pytorch tensor)
            pose: (pytorch tensor)

        """
        if self.mode=='RAM':

            if isinstance(idx,list) or isinstance(idx,np.ndarray):# Positives or negatives
                proj = [self.preprocessing(self.RAM_data['proj'][i]) for i in idx]
                proj = torch.stack(proj,dim=0)
                pose = [self.RAM_data['pose'][i]for i in idx]
                pose = np.stack(pose,axis=0).astype(np.float32)
            else:
                proj = self.preprocessing(self.RAM_data['proj'][idx])
                pose = np.asarray(self.RAM_data['pose'][idx],dtype = np.float32)
        
        else: # Disk
            
            if isinstance(idx,list): # Positives or negatives
                proj = []
                pose = []
                for i in idx:
                    pair = self._load_pose_pcl_pair_(idx)
                    proj.append(self.preprocessing(pair['proj']))
                    pose.append(pair['pose'])

                proj = torch.stack(proj,dim=0)
                pose = np.stack(pose,axis=0).astype(np.float32)
            
            else: # Queries
                pair = self._load_pose_pcl_pair_(idx)
                proj = self.preprocessing(pair['proj'])
                pose = np.asarray(pair['pose'],dtype = np.float32)

        pose = torch.tensor(pose).reshape(-1,2)
        return proj,pose
    


    def get_data(self,anchor_index):
        anchor_index = self.anchor_idx_buffer[anchor_index]
        # point clouds are already converted to the input representation, 
        an_pcl_tns,an_pose_tns = self.get_tuple_data(anchor_index)

        # Get anchor's triplet indices
        triplet_tuple = self.get_triplet_tuple_idx(anchor_index)


        # Load positive data
        pos_idx = triplet_tuple['pos']
        pos_pcl_tns,pos_pose_tns = self.get_tuple_data(pos_idx)

        # Load negative data
        neg_idx = triplet_tuple['neg']
        neg_pcl_tns,neg_pose_tns = self.get_tuple_data(neg_idx)


        pcl_tuple  = {'anchor':an_pcl_tns,'positive':pos_pcl_tns,'negative':neg_pcl_tns}
        pose_tuple = {'anchor':an_pose_tns,'positive':pos_pose_tns,'negative':neg_pose_tns}
      
        return(pcl_tuple,pose_tuple)


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

    def get_GT_Map(self)-> list:
        return ([])

    def __getitem__(self,index:int ):
        
        pcl,pose = self.get_data(index)

        # pcl: {'anchor' , 'positive':[],'negative':[]} 
        # pose: {'anchor' , 'positive':[],'negative':[]} 
        
        return(pcl,pose)

    def __len__(self):
        return(self.num_samples)
        