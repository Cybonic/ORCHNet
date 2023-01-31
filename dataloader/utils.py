


import os
import numpy as np
from tqdm import tqdm


def load_indices(file):
    overlap = []
    for f in open(file):
        f = f.split(':')[-1]
        indices = [int(i) for i in f.split(' ')]
        overlap.append(indices)
    return(np.array(overlap[0]))

def parse_overlap_idx(file):
    idx_array= {}
    for i,(f )in enumerate(open(file)):
        str_array = f.rstrip().split('run:')[-1].split(' ')
        idx = [int(i) for i in str_array]
        if i == 1:
            idx_array['ref']=idx
        else:
            idx_array['query']=idx

    return(idx_array)

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

    anchors   = np.array(anchors,dtype=np.uint32)
    positives = np.array(positives,dtype=np.uint32)
    negatives = np.array(negatives,dtype=np.uint32)

    return anchors,positives,negatives

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

def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([f.split('.')[0] for f in os.listdir(target_dir)])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])

    return(files[idx],fullfiles[idx])

