#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
from scipy.spatial.transform import Rotation as R
import random,math

#def random_pcl_subsampling(points,remissions,max_points):
#  '''
#  https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c

#  '''
#  if not isinstance(points, np.ndarray):
#    points = np.array(points)
#  if not isinstance(remissions, np.ndarray):
#    remissions = np.array(remissions)

#  n_points = points.shape[0]

#  assert n_points>max_points,'Max Points is to big'
#  sample_idx = np.random.randint(0,n_points,max_points)
#  sample_idx  =np.sort(sample_idx)
#  return(points[sample_idx],remissions[sample_idx])

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds


def random_subsampling(points,max_points):
  '''
  https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c

  '''
  if not isinstance(points, np.ndarray):
    points = np.array(points)

  n_points = points.shape[0]

  assert n_points>max_points,'Max Points is to big'
  sample_idx = np.random.randint(0,n_points,max_points)
  sample_idx  =np.sort(sample_idx)
  return(sample_idx)

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, parser = None, max_rem = None, max_points = -1, noise = 0.0002,**argv):

    self.reset()
    self.parser = parser
    self.max_rem = max_rem
    self.max_points = max_points
    self.noise = noise
    self.set_roi_flag = False
    if 'roi' in argv:
      self.set_roi_flag = True
      self.roi = argv['roi']


  def reset(self):
    """ Reset scan members. """
    #self.roi  = None 
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

   
  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    if self.parser != None:
      scan = self.parser.velo_read(filename)
    else: 
      scan = np.fromfile(filename, dtype=np.float32)
      scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    self.set_points(points, remissions)

  def load_pcl(self,scan):
    # Read point cloud already loaded
    self.reset()
   
    points = scan[:, 0:3]    # get xyz
    remissions = np.zeros(scan.shape[0])
    if scan.shape[1]==4:
      remissions = scan[:, 3]  # get remission

    self.set_points(points, remissions)

  def set_points(self, points, remissions):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

  
      # points,remissions = random_pcl_subsampling(points,remissions,self.max_points)

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

  def set_roi(self,value):
    roi = value
    assert isinstance(roi,dict),'roi should be a dictionary'

    PointCloud = self.points.copy()
    Remissions = self.remissions

    if 'xmin' in roi:
      mask = np.where(PointCloud[:, 0] >= roi["xmin"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    
    if 'xmax' in roi:
      mask = np.where(PointCloud[:, 0] <= roi["xmax"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    
    if 'ymin' in roi:
      mask = np.where(PointCloud[:, 1] >= roi["ymin"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    
    if 'ymax' in roi:
      mask = np.where(PointCloud[:, 1] <= roi["ymax"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    
    if 'zmin' in roi:
      min_z = min(PointCloud[:, 2])
      max_z = max(PointCloud[:, 2])
      mask = np.where(PointCloud[:, 2] >= roi["zmin"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    
    if 'zmax' in roi:
      mask = np.where(PointCloud[:, 2] <= roi["zmax"])
      PointCloud = PointCloud[mask]
      Remissions = Remissions[mask]
    

    self.points = PointCloud
    self.remissions = Remissions


  def get_data(self,**argvs):
    
    aug_flag  = False
    norm_flag = False
    
    if self.set_roi_flag:
      #roi = argvs['roi']
      self.set_roi(self.roi)

    import time

    if self.max_points > 0:  # if max_points == -1 do not subsample
      start = time.time()
      #idx = random_subsampling(self.points,self.max_points)
      idx  = fps(self.points, self.max_points)
      end = time.time()
      print(end - start)
      self.points = self.points[idx,:]
      self.remissions = self.remissions[idx]

    if 'norm' in argvs and argvs['norm'] == True:
      self.set_normalization()

    if 'aug' in argvs and argvs['aug'] == True:
      self.set_augmentation()
    
   

    return self.points.astype(np.float32)


  def set_normalization(self):
    norm_pointcloud = self.points - np.mean(self.points, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    self.points = norm_pointcloud

  def set_augmentation(self):
    # https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263

    pointcloud = self.points
    
       # rotation around z-axis
    theta = random.random() * 2. * math.pi # rotation angle
    rot_matrix = np.array([[math.cos(theta), -math.sin(theta),    0],
                          [ math.sin(theta),  math.cos(theta),    0],
                          [0,                             0,      1]])

    rot_pointcloud = rot_matrix.dot(pointcloud.T).T

    # add some noise
    noise = np.random.normal(0,self.noise, (pointcloud.shape))
    noisy_pointcloud = rot_pointcloud + noise

    self.points = noisy_pointcloud
  

    
