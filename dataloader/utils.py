
from .ORCHARDS import ORCHARDS, OrchardDataset
from .KITTI import KITTI,KittiDataset
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




def load_loader(**argv):

    root_dir = get_root_system()
    session = argv['session']


    if argv['dataset'] == 'orchard-uk':
       loader = ORCHARDS(root    = session[root_dir],
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = argv['memory'],
                            max_points    = 30000,
                            **argv
                            )
    else:
        loader = KITTI(root       = session[root_dir],
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = argv['memory'],
                            num_subsamples= argv['subsamples'],
                            max_points = 50000,
                            **argv
                            )
        
    return(loader)