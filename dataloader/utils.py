
from .ORCHARDS import ORCHARDS, OrchardDataset
from .KITTI import KITTI,KittiDataset
from .POINTNETVLAD import POINTNETVLAD
from .FUBERLIN import FUBERLIN
import os

def get_root_system():
    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'
    return(root_dir)

def load_dataset(**argv):

    root_dir = get_root_system()

    root = argv.pop[root_dir]
    if argv['dataset'] == 'orchard-uk':
        loader = OrchardDataset(root =root, **argv)
    else:
        loader = KittiDataset(root =root,**argv)
    return(loader)




def load_dataset(inputs, session,debug=False):

    root_dir = get_root_system()


    if inputs.session == 'kitti':
        
        if debug:
            session['train_loader']['data']['sequence'] = ['00']
            session['val_loader']['data']['sequence'] = ['00']
            print("[Main] Debug mode ON: training and Val on Sequence 00 \n")

        session['val_loader']['data']['modality'] = inputs.modality
        session['val_loader']['data']['sequence'] = inputs.sequence
        session['val_loader']['batch_size'] = inputs.batch_size

        loader = KITTI( root = session[root_dir],
                        train_loader  = session['train_loader'],
                        val_loader    = session['val_loader'],
                        mode          = inputs.memory,
                        sensor        = sensor_cfg,
                        debug         = debug,
                        max_points = 50000)

    elif  inputs.session == 'orchards-uk' :
        
        session['val_loader']['data']['modality'] = inputs.modality
        session['val_loader']['data']['sequence'] = inputs.sequence
        session['val_loader']['batch_size'] = inputs.batch_size

        loader = ORCHARDS(root    = session[root_dir],
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = inputs.memory,
                            sensor        = sensor_cfg,
                            debug         = debug,
                            max_points = 30000)
    
    
    elif  inputs.session == 'pointnetvlad':
        
        session['val_loader']['data']['modality'] = inputs.modality
        session['val_loader']['data']['sequence'] = inputs.sequence
        session['val_loader']['batch_size'] = inputs.batch_size

        loader = POINTNETVLAD(root       = session[root_dir],
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = inputs.memory,
                            max_points = 50000
                            )
    
    elif  inputs.session == 'fuberlin':
        
        #session['train_loader']['root'] =  session[root_dir]
        session['val_loader']['anchor']['root'] =  session[root_dir]
        session['val_loader']['database']['root'] =  session[root_dir]
        session['val_loader']['batch_size'] = inputs.batch_size
        
        loader = FUBERLIN(
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = inputs.memory
                            )
        
        run_name = {'dataset': session['val_loader']['anchor']['sequence']}
    
    return(loader,run_name)