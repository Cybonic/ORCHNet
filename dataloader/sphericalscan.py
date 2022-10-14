#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
from scipy.spatial.transform import Rotation as R
import random,math

class SphericalRangeProjScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self,  H=64, 
                      W=1024, 
                      fov_up=3.0, 
                      fov_down=-25.0,
                      parser = None, 
                      max_depth = None, 
                      image_proj = True, 
                      max_rem = None,
                      noise = 0.0002,
                      roi = {'xmin':-50,'xmax':50,'ymin':-50,'ymax':50,'zmin':0,'zmax':5},
                      **argv):

    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()
    self.parser = parser
    self.max_depth = max_depth # Max range 
    self.max_rem = max_rem
    self.image_proj = image_proj
    self.noise = noise
    self.roi = roi
  
  def reset(self):
    """ Reset scan members. """
    #self.roi  = None 

    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

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

    #PointCloud[:, 2] = PointCloud[:, 2] -  self.roi["zmin"]
    self.points = PointCloud
    self.remissions = Remissions


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
    #theta = random.random() * 2. * math.pi # rotation angle
    rot_matrix = np.array([[ math.cos(target_angle), -math.sin(target_angle),    0],
                          [ math.sin(target_angle),  math.cos(target_angle),    0],
                          [0,                             0,      1]])

    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    # add some noise
    noise = np.random.normal(0,self.noise, (pointcloud.shape))
    noisy_pointcloud = rot_pointcloud + noise
    self.points = noisy_pointcloud

  def set_proj(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)
    depth = np.nan_to_num(depth)

    # get scan components
    scan_x = np.nan_to_num(self.points[:, 0])
    scan_y = np.nan_to_num(self.points[:, 1])
    scan_z = np.nan_to_num(self.points[:, 2])

    # get angles of all points
    #yaw = -np.arctan2(scan_y, scan_x)
    yaw = np.nan_to_num(np.arctan2(scan_y, scan_x))
    pitch = np.arcsin(scan_z / depth)
    pitch = np.nan_to_num(pitch)
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]

    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    if self.image_proj == True:
      self.proj_range =(np.clip(self.proj_range,a_min=0,a_max=self.max_depth)/self.max_depth)*255
      self.proj_range = self.proj_range.clip(max=255).astype(np.uint8)
      
      self.proj_remission = (np.clip(self.proj_remission,a_min=0,a_max=self.max_rem)/self.max_rem)*255
      self.proj_remission = self.proj_remission.clip(max=255).astype(np.uint8)


  def get_data(self,**arg):
    #self.set_normalization()
    if 'aug' in arg and arg['aug']:
      self.set_augmentation()
    #if 'roi' in arg and arg['roi']:
    self.set_roi()
    self.set_proj()

    if arg['modality'] == 'remissions':
      data =  np.expand_dims(np.nan_to_num(self.proj_remission),axis=-1)
    
    elif arg['modality'] == 'points':
      data = self.points
    
    elif arg['modality'] == 'mask':
      data =  np.expand_dims(np.nan_to_num(self.proj_mask),axis=-1)

    elif arg['modality'] == 'range':
      data = np.expand_dims(np.nan_to_num(self.proj_range),axis=-1)
    
    elif arg['modality'] == 'projection' and self.project == True:
      proj_range = np.expand_dims(np.nan_to_num(self.proj_range),axis=-1)
      proj_remission =  np.expand_dims(np.nan_to_num(self.proj_remission),axis=-1)
      data = np.concatenate((proj_range,
                            self.proj_xyz,
                            proj_remission),axis=-1)
    else:
      raise ValueError()
    
    return np.nan_to_num(data)

  def img_proj(self,row,max_value):
    roi = max_value
    if max_value == None:
      roi = row.max()
    return (row*(255/roi)).astype(np.uint8)
  

    
