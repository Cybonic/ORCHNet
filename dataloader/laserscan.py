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





def square_roi(PointCloud,roi_array):
  mask_list = []
  for roi in roi_array:
    region_mask_list = []
    assert isinstance(roi,dict),'roi should be a dictionary'

    if 'xmin' in roi:
      region_mask_list.append((PointCloud[:, 0] >= roi["xmin"]))
    if 'xmax' in roi:
      region_mask_list.append((PointCloud[:, 0]  <= roi["xmax"]))
    if 'ymin' in roi:
      region_mask_list.append((PointCloud[:, 1] >= roi["ymin"]))
    if 'ymax' in roi:
      region_mask_list.append((PointCloud[:, 1]  <= roi["ymax"]))
    if 'zmin' in roi:
      region_mask_list.append((PointCloud[:, 2] >= roi["zmin"]))
    if 'zmax' in roi:
      region_mask_list.append((PointCloud[:, 2]  <= roi["zmax"]))
    
    mask = np.stack(region_mask_list,axis=-1)
    mask = np.product(mask,axis=-1).astype(np.bool)
    mask_list.append(mask)

    
  mask = np.stack(mask_list,axis=-1)
  mask = np.sum(mask,axis=-1).astype(np.bool)
  return(mask)



def cylinder_roi(PointCloud,roi_array):

  mask_list = []
  for roi in roi_array:
    region_mask_list = []
    assert isinstance(roi,dict),'roi should be a dictionary'
    dist = np.linalg.norm(PointCloud[:, 0:3],axis=1) 
    if 'rmin' in roi:
      region_mask_list.append((dist >= roi["rmin"]))
    if 'rmax' in roi:
      region_mask_list.append((dist < roi["rmax"]))
    
    mask = np.stack(region_mask_list,axis=-1)
    mask = np.product(mask,axis=-1).astype(np.bool)
    mask_list.append(mask)
  
  mask = np.stack(mask_list,axis=-1)
  mask = np.sum(mask,axis=-1).astype(np.bool)
  return(mask)




class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, parser = None, max_points = -1, noise = 0.0002, aug=False,**argv):

    self.reset()
    self.parser = parser
    # self.max_rem = max_rem
    self.max_points = max_points
    self.noise = noise
    
    self.set_aug_flag = aug
    
    # Configure ROI
    self.roi = {}
    self.set_roi_flag = False
    if 'square_roi' in argv:
      self.set_roi_flag = True
      self.roi['square_roi'] = argv['square_roi']
    
    if 'cylinder_roi' in argv:
      self.set_roi_flag = True
      self.roi['cylinder_roi'] = argv['cylinder_roi']


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

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)


  # ==================================================================
  def set_roi(self):
    PointCloud = self.points.copy()
    Remissions = self.remissions
    n_points= PointCloud.shape[0]

    mask = np.ones(n_points,dtype=np.bool) # By default use all the points
    # Each roi extraction approach considers the entire point cloud 
    if 'square_roi' in self.roi:
      roi = self.roi['square_roi']
      assert isinstance(roi,list)
      local_mask = square_roi(PointCloud,roi)
      mask = np.logical_and(mask,local_mask)
    
    if 'cylinder_roi' in self.roi:
      roi = self.roi['cylinder_roi']
      assert isinstance(roi,list)
      local_mask = cylinder_roi(PointCloud,roi)
      mask = np.logical_and(mask,local_mask)

    self.points = PointCloud[mask,:]
    self.remissions = Remissions[mask]

  # ==================================================================
  def set_normalization(self):
    norm_pointcloud = self.points - np.mean(self.points, axis=0) 
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    self.points = norm_pointcloud



  # ==================================================================
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
  
  # ==================================================================
  def set_sampling(self):
    import time
    start = time.time()
    idx = random_subsampling(self.points,self.max_points)
    #idx  = fps(self.points, self.max_points)
    end = time.time()
    #print(end - start)
    self.points = self.points[idx,:]
    self.remissions = self.remissions[idx]


  # ==================================================================
  def get_points(self,**argvs):    
    if self.set_roi_flag:
      self.set_roi()

    if self.max_points > 0:  # if max_points == -1 do not subsample
      self.set_sampling()

    #if 'norm' in argvs and argvs['norm'] == True:
    #self.set_normalization()

    if self.set_aug_flag:
      self.set_augmentation()

    return self.points.astype(np.float32),self.remissions.astype(np.float32)


class LaserData(LaserScan):
  def __init__(self, **argv):
    super(LaserData,self).__init__(**argv)
    
    pass


  def get_pcl(self):
    return self.get_points()[0]


  def get_bev_proj(self,parameters):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    points,remissions = self.get_points()
    # Parameters 
    Width  = parameters['W']
    Height = parameters['H'] 
    
    to_image_space = True # parameters['to_image_space'] # Map the depth values to [0,255]
    # Get max and min values along the zz axis
    zmax = self.points[:,2].max()
    zmin = self.points[:,2].min()

    xmax = self.points[:,0].max()
    xmin = self.points[:,0].min()

    ymax = self.points[:,1].max()
    ymin = self.points[:,1].min()


    discretization_x = (xmax - xmin)/(Height)
    discretization_y = (ymax - ymin)/(Width)
    

    # Discretize Feature Map
    PointCloud = np.nan_to_num(np.copy(points))
    Intensity = np.nan_to_num(np.copy(remissions))


    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / discretization_x)  + (Height)/2).clip(min=0,max=Height)
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / discretization_y) + (Width)/2).clip(min=0,max=Height)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height+1, Width+1))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    xx_idx = np.int_(PointCloud_frac[:, 0])
    yy_idx = np.int_(PointCloud_frac[:, 1])

    heightMap[xx_idx,yy_idx] = PointCloud_frac[:, 2]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height+1, Width+1))
    densityMap = np.zeros((Height+1, Width+1))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    
    PointCloud_top = PointCloud[indices]
    Intensity_top = Intensity[indices]

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = Intensity_top

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    if to_image_space == True:
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
      max_height = float(np.abs(local_min_height - local_max_height))

      heightMap = np.clip(heightMap,a_max = zmax,a_min = zmin)
      heightMap = ((heightMap-local_min_height)/max_height)*255
      heightMap = np.clip(heightMap,a_min=0,a_max=255).astype(np.uint8)
    
    density = np.expand_dims(densityMap,axis=-1)
    height  = np.expand_dims(heightMap,axis=-1)
    intensity = np.expand_dims(intensityMap,axis=-1)

    return {'height': height, 'density': density, 'intensity': intensity}
    
  

  def get_shperical_proj(self,parameters):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # Get the required parameters
    proj_fov_up = parameters['fov_up']
    proj_fov_down = parameters['fov_down']
    proj_W = parameters['W']
    proj_H = parameters['H']
    max_rem = parameters['max_rem']
    max_depth = parameters['max_depth']
    to_image_space = True # parameters['to_image_space'] # Map the depth values to [0,255]

    points,remissions = self.get_points()

    # projected range image - [H,W] range (-1 is no data)
    proj_range = np.full((proj_H, proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    proj_remission = np.full((proj_H, proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((proj_H, proj_W),
                              dtype=np.int32)       # [H,W] mask


    
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)
    depth = np.nan_to_num(depth)

    # get scan components
    scan_x = np.nan_to_num(points[:, 0])
    scan_y = np.nan_to_num(points[:, 1])
    scan_z = np.nan_to_num(points[:, 2])

    # get angles of all points
    #yaw = -np.arctan2(scan_y, scan_x)
    yaw = np.nan_to_num(np.arctan2(scan_y, scan_x))
    pitch = np.arcsin(scan_z / depth)
    pitch = np.nan_to_num(pitch)
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]

    depth = depth[order]
    indices = indices[order]

    points = self.points[order]
    remission = remissions[order]
    
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_remission[proj_y, proj_x] = remission
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)

    if to_image_space == True:
      proj_range =(np.clip(proj_range,a_min=0,a_max=max_depth)/max_depth)*255
      proj_range = proj_range.clip(max=255).astype(np.uint8)
      
      proj_remission = (np.clip(proj_remission,a_min=0,a_max=max_rem)/max_rem)*255
      proj_remission = proj_remission.clip(max=255).astype(np.uint8)

    # Expand dim to create a matrix [H,W,C]
    proj_range = np.expand_dims(proj_range,axis=-1)
    proj_remission = np.expand_dims(proj_remission,axis=-1)
    proj_idx = np.expand_dims(proj_idx,axis=-1)
    proj_mask = np.expand_dims(proj_mask,axis=-1)

    return {'range': proj_range, 'xyz': proj_xyz, 'remission': proj_remission,'idx':proj_idx,'mask': proj_mask}


  def get_data(self,modality,param):
    
    if modality == 'pcl':
      out = self.get_pcl()
    elif modality in ['bev']:
      out = self.get_bev_proj(param)
      out = np.concatenate(list(out.values()),axis=-1)
    elif modality in ['range']:
      out = self.get_shperical_proj(param)
      out = np.concatenate(list(out.values()),axis=-1)

    return out





  

    
