

import numpy as np

from scipy.spatial.transform import Rotation as R
import random,math
# https://github.com/Cybonic/Complex-YOLOv3/blob/master/utils/kitti_bev_utils.py

class BirdsEyeViewScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self,  H=64, 
                      W=1024, 
                      parser = None, 
                      image_proj=True, 
                      noise = 0.0002, 
                      roi = {'xmin':-50,'xmax':50,'ymin':-50,'ymax':50,'zmin':0,'zmax':5},
                      **argv):

    self.proj_H = H
    self.proj_W = W
    self.reset()
    self.parser = parser
    self.roi = roi
    self.image_proj = image_proj
    self.noise = noise
    self.discretization = (roi["xmax"] - roi["xmin"])/self.proj_H
  
  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.intensity = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)
    self.density = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    self.height = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

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

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

  def set_roi(self):
    roi =self.roi 

    assert 'xmin' in roi
    assert 'xmax' in roi
    assert 'ymin' in roi
    assert 'ymax' in roi
    assert 'zmin' in roi
    assert 'ymin' in roi
    assert 'ymax' in roi

    PointCloud = self.points.copy()
    Remissions = self.remissions
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= self.roi["xmin"]) & (PointCloud[:, 0] <=  self.roi["xmax"]) & (PointCloud[:, 1] >=  self.roi["ymin"]) & (
            PointCloud[:, 1] <=  self.roi["ymax"]) & (PointCloud[:, 2] >=  self.roi["zmin"]) & (PointCloud[:, 2] <=  self.roi["zmax"]))
            
    PointCloud = PointCloud[mask]
    Remissions = Remissions[mask]
    
    #depth = np.linalg.norm(PointCloud, 2, axis=1)
    #PointCloud = PointCloud[depth>1,:]
    #Remissions = Remissions[depth>1]

    PointCloud[:, 2] = PointCloud[:, 2] -  self.roi["zmin"]
    self.points = PointCloud
    self.remissions = Remissions



  def set_proj(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """

    Height = self.proj_H + 1
    Width = self.proj_W + 1

    # Discretize Feature Map
    PointCloud = np.nan_to_num(np.copy(self.points))
    Intensity = np.nan_to_num(np.copy(self.remissions))

    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / self.discretization) + Height / 2)
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / self.discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    
    PointCloud_top = PointCloud[indices]
    Intensity_top = Intensity[indices]

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = Intensity_top

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    if self.image_proj:
      # Density
      densityMap = densityMap*255
      densityMap = np.clip(densityMap,a_min=0,a_max=255).astype(np.uint8)
      # Intensity
      intensityMap = (np.clip(intensityMap,a_min = 0, a_max = 100)/100) *255
      intensityMap = np.clip(intensityMap,a_min=0,a_max=255).astype(np.uint8)
      # Height
      local_max_height = heightMap.max()
      local_min_height = heightMap.min()

      #self.max_height = float(np.abs(self.roi['zmax'] - self.roi['zmin']))
      self.max_height = float(np.abs(local_min_height - local_max_height))

      heightMap = np.clip(heightMap,a_max = self.roi['zmax'],a_min = self.roi['zmin'])
      heightMap = ((heightMap-local_min_height)/self.max_height)*255
      heightMap = np.clip(heightMap,a_min=0,a_max=255).astype(np.uint8)
    
    self.density = np.expand_dims(densityMap,axis=-1)
    self.height  = np.expand_dims(heightMap,axis=-1)
    self.intensity = np.expand_dims(intensityMap,axis=-1)


  def set_normalization(self):
    norm_pointcloud = self.points - np.mean(self.points, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    self.points = norm_pointcloud

  def set_augmentation(self):
    # https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
    pointcloud = self.points
    # rotation around z-axis
    rotation_space = np.array([0,math.pi])
    target_angle = np.random.randint(0,len(rotation_space),1).item()
    theta = rotation_space[target_angle]
    #theta = random.random() * 2. * math.pi # rotation angle
    rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                          [ math.sin(theta),  math.cos(theta),    0],
                          [0,                             0,      1]])

    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    # add some noise
    noise = np.random.normal(0,self.noise, (pointcloud.shape))
    noisy_pointcloud = rot_pointcloud + noise

    self.points = noisy_pointcloud

  def get_data(self,**arg):
    #self.set_normalization()
    if 'aug' in arg and arg['aug']:
      self.set_augmentation()
    #if 'roi' in arg and arg['roi']:
    self.set_roi()

    self.set_proj()

    if arg['modality'] == 'intensity':
        data = np.nan_to_num(self.intensity)
    elif arg['modality']  == 'density':
        data = np.nan_to_num(self.density)
    elif arg['modality']  == 'height':
        data = np.nan_to_num(self.height)
    elif arg['modality']  == 'bev':
        data = np.concatenate((self.intensity,self.height,self.density),axis=-1)
    
    return np.nan_to_num(data)

