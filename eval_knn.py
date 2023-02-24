#!/usr/bin/env python3

import argparse
import yaml
from shutil import copyfile
import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import torch 
from tqdm import tqdm
from networks.orchnet import *

from dataloader.ORCHARDS import ORCHARDS

from networks import model
from utils.retrieval import retrieval_knn
from utils.metric import retrieve_eval

def load_dataset(inputs,session,max_points=50000,debug=False):

    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'

        
    session['val_loader']['data']['modality'] = inputs.modality
    session['val_loader']['data']['sequence'] = inputs.sequence
    session['val_loader']['batch_size'] = inputs.batch_size

    loader = ORCHARDS(root    = session[root_dir],
                        train_loader  = session['train_loader'],
                        test_loader    = session['val_loader'],
                        mode          = inputs.memory,
                    )
                        #sensor        = sensor_cfg,
                        #debug         = debug,
                        #max_points = 30000)

    return(loader.get_val_loader())


class PlaceRecognition():
    def __init__(self,model,loader,top_cand,windows,eval_metric,device):

        self.eval_metric = eval_metric
        self.model  = model.to(device)
        self.loader = loader
       
        self.device = device
        self.top_cand = top_cand
        self.windows = windows

        # Eval data
        try:
            self.database = loader.dataset.get_idx_universe()
            self.anchors = loader.dataset.get_anchor_idx()
            table = loader.dataset.get_GT_Map()
            poses = loader.dataset.get_pose()
            
        except: 
            self.database = loader.dataset.dataset.get_idx_universe()
            self.anchors = loader.dataset.dataset.get_anchor_idx()
            table = loader.dataset.dataset.get_GT_Map()
            poses = loader.dataset.dataset.get_pose()

        self.true_loop = np.array([np.where(line==1)[0] for line in table])
        

    def get_descriptors(self):
        return self.descriptors
        
    def run(self):
        
        assert isinstance(self.top_cand,list)

        
        self.descriptors = self.generate_descriptors(self.model,self.loader)
        # None Retrieval Area
        pred_loops = []
        target_loops= []
        target_loops = self.true_loop[self.anchors]
        descriptor_idx = list(self.descriptors.keys())
        # Compute number of samples to retrieve correspondin to 1% 
        database_size = len(self.database)- self.windows
        one_percent = int(round(database_size/100,0))
        # Append 1% to candidates to retrieve
        self.top_cand.append(one_percent)
        # Get the biggest value
        self.max_top = max(self.top_cand)
        for anchor in tqdm(self.anchors,"Retrivel"):
            database_idx = self.database[:anchor-self.windows] # 
            # Split descriptors
            query_dptrs =  np.array([self.descriptors[i] for i in [anchor] if i in descriptor_idx ])
            map_dptrs = np.array([self.descriptors[i] for i in database_idx if i in descriptor_idx ])
            # Retrieve loops 
            retrieved_loops ,scores = retrieval_knn(query_dptrs, map_dptrs, top_cand = self.max_top, metric = self.eval_metric)
            #if len(retrieved_loops)==0:
            #    continue
            pred_loops.append(retrieved_loops[0])
        # Evaluate retrieval
        pred_loops = np.array(pred_loops)
        target_loops = np.array(target_loops)

        overall_scores = {}
        for top in self.top_cand:
            scores = retrieve_eval(pred_loops,target_loops, top = top)
            overall_scores[top]=scores
        # Post on tensorboard
        return overall_scores



    def generate_descriptors(self,model,loader):
            model.eval()
            dataloader = iter(loader)
            num_sample = len(loader)
            tbar = tqdm(range(num_sample), ncols=100)

            #self._reset_metrics()
            prediction_bag = {}
            idx_bag = []
            for batch_idx in tbar:
                input,inx = next(dataloader)
                input = send_to_device(input,self.device)
                # Generate the Descriptor
                prediction = model(input)
                assert prediction.isnan().any() == False
                # Keep descriptors
                for d,i in zip(prediction.detach().cpu().numpy().tolist(),inx.detach().cpu().numpy().tolist()):
                    prediction_bag[int(i)] = d
            return(prediction_bag)



def send_to_device(input,device):
    output = []
    if isinstance(input,list):
        for item in input:
            output_dict = {}
            for k,v in item.items():
                value = v.to(device)
                output_dict[k]=value
            output.append(output_dict)

    elif isinstance(input,dict):
        output = {}
        for k,v in input.items():
            value = v.to(device)
            output[k]=value
    else:
        output = input.to(device)
    #value = value.cuda(non_blocking=True)
    return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default='VLAD_pointnet',
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=False,
        default='SamplingOrchardsvSameDataset/P1-N20-NO-TNET_F64/MP-30000/LazyQuadrupletLoss L2/autumn',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--cfg', '-f',
        type=str,
        required=False,
        default='sensor-cfg',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--resume', '-p',
        type=str,
        required=False,
        default='checkpoints/FITTING/LazyQuadrupletLoss_L2/autumn/VLAD_pointnet/best_model.pth',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--memory',
        type=str,
        required=False,
        default='Disk',
        choices=['Disk','RAM'],
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--debug', '-b',
        type=bool,
        required=False,
        default=False,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--plot',
        type=int,
        required=False,
        default=1,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--modality',
        type=str,
        required=False,
        default='pcl',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default='orchard-uk',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        required=False,
        default='autumn',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=10,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--max_points',
        type=int,
        required=False,
        default = 500,
        help='sampling points.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    # open arch config file
    cfg_file = os.path.join('dataloader','sensor-cfg.yaml')
    print("Opening data config file: %s" % cfg_file)
    sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))


    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    SESSION['model']['type'] = FLAGS.model
    print("----------")
    print("INTERFACE:")
    print("Root: ", SESSION['root'])
    #print("Dataset  -> Validation: ", SESSION['val_loader']['data']['dataset'])
    #print("Sequence -> Validation: ", SESSION['val_loader']['data']['sequence'])
    print("Memory: ", FLAGS.memory)
    print("Model:  ", FLAGS.model)
    print("Debug:  ", FLAGS.debug)
    print("Resume: ", FLAGS.resume)
    print(f'Device: {FLAGS.device}')
    print(f'batch size: {FLAGS.batch_size}')
    print("----------\n")

    
    dataloader = load_dataset(FLAGS,SESSION)
                                
    ###################################################################### 
    SESSION['train_loader']['data']['max_points'] = FLAGS.max_points
    SESSION['val_loader']['data']['max_points'] = FLAGS.max_points
    modality = FLAGS.modality + '_param'

    SESSION[modality]['max_samples'] = FLAGS.max_points # For VLAD one as to define the number of samples
    model_ = model.ModelWrapper(**SESSION['model'],loss= [], **SESSION[modality])
  
    #run_name['model'] = FLAGS.model
    #run_name['experiment'] = FLAGS.experiment
  
    SESSION['retrieval']['top_cand'] = list(range(1,25,1))

    pl = PlaceRecognition(model_,dataloader,25,'L2',FLAGS.device)
    
    results = pl.run()

    descriptors = pl.get_descriptors()
    #results,descriptors = trainer.Eval()

    dataset =FLAGS.session
    
    columns = ['top','recall']
    values = [v['recall'] for v in list(results.values())]

    rows = [[t,v] for t,v in zip(list(results.keys()),values)]
    import pandas as pd


    df = pd.DataFrame(rows,columns = columns)
    top = rows[0][0]
    score = round(rows[0][1],2)
    
    results_dir = os.path.join('predictions',dataset,'place','results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    file_results = os.path.join(results_dir,f'{FLAGS.modality}-{FLAGS.model}_{score}@{top}.csv')

    df.to_csv(file_results)

    descriptors_dir = os.path.join('predictions',f'{dataset}','place','descriptors')
    if not os.path.isdir(descriptors_dir):
        os.makedirs(descriptors_dir)

    file_name = os.path.join(descriptors_dir,f'{FLAGS.modality}-{FLAGS.model}_{score}@{top}.npy')
    torch.save(descriptors,file_name)

 

  
  