#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import matplotlib as mpl
#import config as cnf

def step_color_fader(colors,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    mix_shape = mix.shape[0]
    #mix = mix
    mix_idx = np.argsort(mix)
    n_steps = len(colors)
    color_points = int(mix_shape/n_steps)

    steps = list(range(0,mix_shape+1,color_points))

    c1 = np.array(mpl.colors.to_rgb(colors[0])).reshape(1,-1)
    c1= np.tile(c1,(mix_shape,1))

    for i,c in enumerate(colors):
      c_color = np.array(mpl.colors.to_rgb(c))
      c1[mix_idx[steps[i]:steps[i+1]]] = c_color

    return [mpl.colors.to_hex(c) for c  in c1]

def gradient_color_fader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    mix_shape = mix.shape[0]
    mix = mix.reshape(-1,1)
    
    c1 = np.array(mpl.colors.to_rgb(c1)).reshape(1,-1)
    c1= np.tile(c1,(mix_shape,1))
    
    c2 = np.array(mpl.colors.to_rgb(c2)).reshape(1,-1)
    c2= np.tile(c2,(mix_shape,1))

    c_vector = (1-mix)*c1 + mix*c2
    c_vector = np.clip(c_vector,a_min=0,a_max=1)
    return [mpl.colors.to_hex(c) for c  in c_vector]


def colorize_pcl(points):
  z = points[:,2]
  c1='yellow' #blue
  c2='blue' #green
  
  z0 = z-z.min()
  z_nom = z0/z0.max()
  #color = gradient_color_fader('red','yellow',z_nom)
  color = step_color_fader(['red','green','yellow'],z_nom)
  return color


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self,dataset,size= 10 ):
    
    #self.scan = scan
    self.size = size
    self.pointcloud = dataset
    self.offset = 0
    self.total = len(dataset)

    self.reset()
    self.update_scan()
    # make instance colors

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "next"  # no, next, back, quit are the possibilities
    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    self.canvas.show()
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)

    #self.img_view = vispy.scene.widgets.ViewBox(border_color='blue', parent=self.canvas.scene)
    #self.pose_view = vispy.scene.widgets.ViewBox(
    #    border_color='blue', parent=self.canvas.scene)

    self.grid.add_widget(self.scan_view, 0, 0)
    #self.grid.add_widget(self.img_view, 0, 1)

    self.scan_vis = visuals.Markers()
    #self.image_vis = visuals.Image()
    #self.poses_vis = visuals.Markers()
    
    self.scan_view.camera =  'turntable'
    #self.img_view.camera = 'turntable'
    
    self.scan_view.add(self.scan_vis)
    #self.img_view.add(self.image_vis)
    #self.pose_view.add(self.poses_vis)

    visuals.XYZAxis(parent=self.scan_view.scene)
    #visuals.XYZAxis(parent=self.pose_view.scene)

 
  def update_scan(self):
    # first open data
    # then change names
    title = "scan "
    self.canvas.title = title
    points,scan_name = self.pointcloud._get_proj_(self.offset)
    
    
    color = colorize_pcl(points)
    self.scan_vis.set_data(points,size = self.size, face_color = color )
    #self.poses_vis.set_data(poses,size = 10)

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
  
  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()


  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()



