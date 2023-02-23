#!/usr/bin/env python3

import yaml
from shutil import copyfile
import os
from tqdm import tqdm
import sys
import argparse
import numpy as np
from dataloader.ORCHARDS import OrchardDataset
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio
from PIL import Image,ImageOps



if __name__ == '__main__':

  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
    '--root',
    type=str,
    required=False,
    # default='checkpoints/range-rerecord_sparce-AttVLAD_resnet50-0.87.pth',
    #default='/media/tiago/BIG',
    default='predictions',
    help='Directory to get the trained model.'
  )
  parser.add_argument(
    '--dataset',
    type=str,
    required=False,
    default='orchard-uk',
    #default='kitti',
    help='Directory to get the trained model.'
  )

  parser.add_argument(
    '--sequence',
    type=str,
    required=False,
    #default='02',
    default='autumn',
    help='Directory to get the trained model.'
  )

  parser.add_argument(
    '--max_points',
    type=str,
    required=False,
    default=20000,
    help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()

  # open arch config file
  cfg_file = os.path.join('dataloader','sensor-cfg.yaml')

  print("Opening data config file: %s" % cfg_file)
  #sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))

  session = FLAGS.dataset + '.yaml'
  session_cfg_file = os.path.join('sessions', session)
 
  print("Opening session config file: %s" % session_cfg_file)
  SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
 
  print("----------")
  print("INTERFACE:")
  print("Root: ", SESSION['root'])
  print("Dataset  : ", FLAGS.dataset)
  print("Sequence : ",FLAGS.sequence)
  print("Max points : ",FLAGS.max_points)
  print("----------\n")

  root = SESSION['root']
  dataset    = FLAGS.dataset
  sequence   = FLAGS.sequence
  max_points = FLAGS.max_points

  loader = OrchardDataset(root,'',sequence,sync = True,modality = 'bev',square_roi = [{'xmin':-15,'xmax':15,'ymin':-15,'ymax':15,'zmax':1}]) #cylinder_roi=[{'rmax':10}])

  fig = Figure(figsize=(5, 4), dpi=100,)
  fig, ax = plt.subplots()

  filename = 'projection.gif'
  canvas = FigureCanvasAgg(fig)
  writer = imageio.get_writer(filename, mode='I')

  fig, ax = plt.subplots(1, 1)
  num_samples = len(loader)
  import cv2
  import numpy as np




  for i in tqdm(range(500,num_samples,10)):
    
    input = loader._get_modality_(i)[:,:,0] # Get only Bev projection
    input_im = input # .astype(np.uint8)
    #input_im[input_im<100] = 0
    pil_range = Image.fromarray(input_im.astype(np.uint8))
    #cv2.imwrite("test.png", input_im)
    #

    #input_im  = T.ToTensor()(input_im).unsqueeze(0)
    #pil_range = T.ToPILImage()(input_im[0])
    
   
    pil_range = ImageOps.colorize(pil_range, black="white", white="black")
    #pil_range = ImageOps.autocontrast(pil_range,ignore=0)
    #pil_range.show()
    #pil_range.save('test.png')
    #plt.show()
    X = np.asarray(pil_range)
    writer.append_data(X)






  
  