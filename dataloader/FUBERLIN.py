import os 
from tqdm import tqdm 
import numpy as np
import yaml
from .sphericalscan import SphericalRangeProjScan
from .birdsviewscan import BirdsEyeViewScan
from torch.utils.data import DataLoader
import torchvision.transforms as Tr
from .laserscan import LaserScan
from torch.utils.data import DataLoader

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

class velo_parser():
    def __init__(self):
        self.dt = []

    def velo_read(self,scan_path):
        #print(scan_path)
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return(np.array(scan))


def comp_score_table(anchor,database):
    '''
    
    '''
    if not isinstance(anchor,np.ndarray):
        anchor = np.array(anchor)
    
    if not isinstance(database,np.ndarray):
        database = np.array(database)

    table_height = anchor.shape[0]
    table_width = database.shape[0]
    
    table = np.zeros((table_height,table_width),dtype=np.float32)
    table = []
    for i in range(table_height):
        qdistance = np.linalg.norm(anchor[i,:]- database,axis=1)
        table.append(qdistance.tolist())
    return(np.asarray(table))


def comp_gt_table(anchor,database,pos_thres):
    '''
    
    computes a matrix of size [len(anchor) x len(database)] of relative distances
    
    return a matrix with '1' where the corresponding distance is < pos_thres 
    '''
    dist_table  = comp_score_table(anchor,database)
    loop_table = np.zeros(dist_table.shape)
    loop_table[dist_table<pos_thres] = 1
    # Print positive distances
    # print(dist_table[loop_table.astype(np.bool8)])
    
    return(loop_table)

def read_sync_file(file):
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

class agent():
    def __init__(self,root,sequence,name,modality, max_points, sync=True, aug = False, **argv):
        self.verbose = 1
        self.name = name
        self.modality = modality
        self.aug = aug
        self.dir = os.path.join(root,sequence,name)
        assert os.path.isdir(self.dir)
        # Load Pose data
        self.pose_file = os.path.join(self.dir,'pose.txt')
        assert os.path.isfile(self.pose_file), 'Pose file does not exist: ' + self.pose_file
        self.pose = load_pose_to_RAM(self.pose_file)
        
        # Load PCL files
        self.pcl_dir = os.path.join(self.dir,'point_cloud')
        assert os.path.isdir(self.pcl_dir)
        self.pcl_files = np.array([os.path.join(self.pcl_dir,file) for file in os.listdir(self.pcl_dir)])
        
        if sync == True:
            self.sync_data()

        
        cfg_file = os.path.join('dataloader','fuberlin-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        
        dataset_param = sensor_cfg[name]
        
        sensor =  sensor_cfg[dataset_param['sensor']]

        if modality in ['range','projection','remissions']:
            proj_pram = dataset_param['RP']
            self.proj = SphericalRangeProjScan(**sensor,**proj_pram,roi = dataset_param['roi'],parser = velo_parser(),**argv)
        elif modality in ['intensity','density','height','bev']:
            proj_pram = dataset_param['BEV']
            self.proj = BirdsEyeViewScan(**proj_pram, roi = dataset_param['roi'], parser = velo_parser(),**argv)
        else:
            self.proj = LaserScan(parser = velo_parser(), max_points = max_points, **argv)


    def sync_data(self):
        # Sync pose data
        pose_sync_file = os.path.join(self.dir,'sync_pose_idx.txt')
        assert os.path.isfile(pose_sync_file), 'Pose sync File does not exist: ' + self.pose_sync_file
        sync_pose_idx = read_sync_file(pose_sync_file)
        self.sync_pose =  self.pose[sync_pose_idx]

        # sync pcl data
        pcl_sync_file = os.path.join(self.dir,'sync_point_cloud_idx.txt')
        assert os.path.isfile(pcl_sync_file),'PCL sync File does not exist: ' + self.pcl_sync_file
        sync_plc_idx = read_sync_file(pcl_sync_file)
        self.sync_pcl_files = self.pcl_files[sync_plc_idx]

        if self.verbose:
            print("\nSync Data")
            print("Pose points = " + f'%4d'%(len(self.pose)) + ' --> ' +  f'%4d'%(len(self.sync_pose)))
            print("PCL  frames = " +  f'%4d'%(len(self.pcl_files)) + ' --> ' +  f'%4d'%(len(self.sync_pcl_files))) 
        
            
    def __str__(self):
        return self.dir + '\n' + "pcl count: " + f'%4d'%(len(self.pcl_files)) + ' ' + "pose count: " + f'%4d'%(len(self.pose))
    
    def get_pose(self):
        return self.sync_pose
    
    def get_pcl_files(self):
        return self.sync_pcl_files
    
    def __len__(self):
        return(len(self.sync_pose))

    def __call__(self,idx):
        file = self.sync_pcl_files[idx]
        self.proj.open_scan(file)
        return self.proj.get_data(modality = self.modality, aug = self.aug)




class FUBerlinDataset():
    def __init__(self,  anchor_parm: dict,
                        database_parm: dict,
                        memory: str
                        ):

       
        #print(self.anchors)
        self.anchors = agent(**anchor_parm)
        #print(self.anchors)
        self.database = agent(**database_parm)
    
    
    def __str__(self):
        return 'Anchors:\n' + str(self.anchors)  +  '\n\nDatabase:\n' + str(self.database) + '\n\n'
    
    def get_anchor_pose(self):
        return self.anchors.get_pose()
    
    def get_database_pose(self):
        return self.database.get_pose()


class FUBerlinEval( ):
    def __init__(self,  
            anchor_parm:dict,
            database_parm:dict,
            memory: str,
            **argv
            ):

        self.preprocessing = PREPROCESSING

        self.anchors = agent(**anchor_parm)
        self.database = agent(**database_parm)
        self.pcl_files_collection = np.concatenate((self.anchors.get_pcl_files(),self.database.get_pcl_files()))
        self.pose_collection = np.concatenate((self.anchors.get_pose(),self.database.get_pose()))
        
        anchor_len = len(self.anchors)
        database_len = len(self.database)

        self.anchors_idx  = np.arange(0,anchor_len)
        self.database_idx = np.arange(anchor_len,anchor_len + database_len)
        self.idx_universe = np.arange(0,anchor_len + database_len)
        # self.colection_pcl_file = np.stack((,self.anchors.get_pcl_files()),axis=0)
        self.memory = memory
        #self.preprocessing = PREPROCESSING
        self.gt_table  = comp_gt_table(self.anchors.get_pose(),self.database.get_pose(),15)
        
        if self.memory=='RAM':
            # Load data to RAM
            self.pcl_collection = self.load_RAM()

    def load_RAM(self):
        img   = {} 
        k = 0     
        len_anchor = len(self.anchors)
        for i in tqdm(range(len_anchor),"Loading Anchor to RAM"):
            img[k] = self.anchors(i)#.astype(np.uint8)
            k+=1

        len_database = len(self.database)
        for j in tqdm(range(len_database),"Loading Database to RAM"):
            img[k] = self.database(j)#.astype(np.uint8)
            k+=1

        return img
    
    def __len__(self):
        return len(self.pcl_collection)

    def get_map_idx(self):
        return self.database_idx

    def get_anchor_idx(self):
        return self.anchors_idx

    def get_GT_Map(self):
        return self.gt_table
    
    def get_data(self,idx):
        if self.memory == 'RAM':
            data = self.pcl_collection[idx]         
        
        elif self.memory == 'Disk':
            if idx in self.anchors_idx:
                data = self.anchors(idx)
            elif idx in self.database_idx:
                data = self.database(idx)
        return data

    def __call__(self,idx):
        global_index = idx # Only useful when subsampler is on
        data = self.get_data(global_index)
        return(data)

    def __getitem__(self,idx):
        global_index = idx
        data = self.get_data(global_index)
        plc = self.preprocessing(data)
        return(plc,global_index)


class FUBERLIN():
    def __init__(self,  
            val_loader: dict,
            **argv
            ):
        
        
        anchor_parm = val_loader['anchor']
        database_parm = val_loader['database']
        memory = val_loader['memory']
        batch_size = val_loader['batch_size']

        self.loader = FUBerlinEval(anchor_parm,database_parm,memory)

        self.valloader   = DataLoader(self.loader,
                                    batch_size = batch_size,
                                    num_workers= 0,
                                    pin_memory=False,
                                    )
    def get_val_loader(self):
        return self.valloader
    
    def get_train_loader(self):
        return []

