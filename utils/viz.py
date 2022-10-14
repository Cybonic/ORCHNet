import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio

import matplotlib.colors as mcolors


class myplot():
    def __init__(self,delay = 0.01,offset=20):
        self.fig = Figure(figsize=(5, 4), dpi=100,)
        self.fig, self.ax = plt.subplots()
        self.gif_record_flag = False
        self.delay  = delay
        self.offset = offset

    def record_gif(self, filename = 'path.gif'):
        self.gif_record_flag = True
        self.canvas = FigureCanvasAgg(self.fig)
        self.writer = imageio.get_writer(filename, mode='I')

    def init_plot(self,x,y,s,c):
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        color = mcolors.CSS4_COLORS[c]
        self.p = self.ax.scatter(x,y,s = s, c = color)
    
    def update_color_scale(self,color=[],scale = []):
        if color != []:
            self.p.set_color(color)

        if scale != []:
            self.p.set_sizes(scale)
        self.fig.canvas.draw()
        self.ax.set_aspect('equal')
        plt.pause(self.delay)

    def update_plot(self,x,y,offset=20,color=[],zoom=10,scale = []):
        self.p.set_offsets(np.c_[x,y])

        if color != []:
            self.p.set_color(color)

        if scale != []:
            self.p.set_sizes(scale)

        if len(x)<zoom:
            xmax,xmin= max(x),min(x)
            ymax,ymin = max(y),min(y)
        else: 
            xmax,xmin= max(x[-zoom:]),min(x[-zoom:])
            ymax,ymin = max(y[-zoom:]),min(y[-zoom:])

        self.ax.axis([xmin - offset, xmax + offset, ymin - offset, ymax + offset])
        self.ax.set_aspect('equal')
        self.fig.canvas.draw()
        plt.pause(self.delay)
        
        if self.gif_record_flag == True:
            buf = self.fig.canvas.buffer_rgba()
            X = np.asarray(buf)
            self.writer.append_data(X)

    def play(self,x,y,s=10,c='g',zoom=10,warmup=50):
        self.init_plot(x[:warmup],y[:warmup],s,c)
        for i,(xx,yy) in enumerate(zip(x[warmup:],y[warmup:])):
            if i%self.offset==0:
                self.update_plot(x[:warmup+i],y[:warmup+i],zoom=zoom)

    def static_plot(self,x,y,s=20,c='g',label = ''):
        self.p = self.ax.scatter(x,y,s = s, c = c,label = label)
        self.ax.set_aspect('equal')
        self.fig.canvas.draw()
        if label != '':
            self.ax.legend()
    
    def show(self):
        plt.show()
    
    def xlabel(self,input):
        plt.xlabel(input)
    
    def ylabel(self,input):
        plt.ylabel(input)


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def color_retrieval_on_map(num_samples,anchors,tp,fp):
    
    c = np.array(['k']*num_samples)
    c[anchors] = 'b'
    c[tp] = 'g'
    c[fp] = 'r'

    s = np.ones(num_samples)*15
    s[tp]     = 150
    s[anchors]= 50
    s[fp]     = 50

    return(c,s)

def plot_retrieval_on_map(pose,anchors,tp,fp,name):
    """
    Input Args:
     - name: (str) Name of image to be saved
     - pose: (numpy) nx3 array of poses
     - anchors: (numpy) list of anchor indices
     - tp: (numpy) list of true positive indices
     - fp: (numpy) list of false positive indices
    
    Output arg:
     - Numpy array of the image

    """
    fig, ax = plt.subplots()
    canvas = FigureCanvasAgg(fig)
    num_samples = pose.shape[0]
    
    all_idx = np.arange(num_samples)
    
    top_pos = []
    for pos in tp:
        top_pos.extend(np.unique(pos))
    top_pos = np.unique(top_pos)

    
    green_idx = np.setxor1d(np.setxor1d(all_idx,anchors),top_pos)
    
    c = np.array(['k']*num_samples)
    c[anchors] = 'b'
    c[top_pos] = 'g'
    c[top_pos] = 'r'

    s = np.ones(num_samples)*30
    s[top_pos] = 30
    s[anchors] = 10
    s[fp]      = 50


    ax.scatter(pose[green_idx,0],pose[green_idx,1],s = s[green_idx], c = c[green_idx],label='path')
    ax.scatter(pose[top_pos,0],pose[top_pos,1],s = s[top_pos], c = c[top_pos],label = 'positive')
    ax.scatter(pose[anchors,0],pose[anchors,1],s = s[anchors], c = c[anchors],label = 'anchor')
    plt.xlabel('m')
    plt.ylabel('m')
    ax.legend()
    #p = ax.scatter(pose[:,0],pose[:,1],s = s, c = c)
    ax.set_aspect('equal')
    canvas.draw()
    
    plt.savefig(name)
    #buf = canvas.buffer_rgba()
    #X = np.asarray(buf)
    return ax


def color_similarity_on_map(similarity,query_idx,plot_pos = False):
    num_samples = len(similarity)
    c = np.array([mpl.colors.to_hex('Black')]*num_samples)
    scale = np.ones(num_samples)*15

    norm_sim = (similarity-similarity.min())/(similarity-similarity.min()).max()
    
    range_idx = list(range(similarity.shape[0]))
    pos_idx = np.argmin(norm_sim)
    c1='yellow' #blue
    c2='red' #green
    
    for s,i in zip(norm_sim,range_idx):
        color=colorFader(c1,c2,s)
        c[i] = color
        scale[i] = 50
        
    scale[query_idx] = 90
    scale[pos_idx] = 200
    c[query_idx] = mpl.colors.to_hex('green')
    c[pos_idx] = mpl.colors.to_hex('blue')

    return c,scale



def color_scale_similarity_on_map(pose,query,map,similarity):
    """
    Input args:
     - name: (str) Name of image to be saved
     - pose: (numpy) nx3 array of poses
     - query:  query indice
     - similarity: (numpy) array of similarities, with the same order as pose
    
    Output arg:
     - Numpy array of the image
    """
    # https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    fig, ax = plt.subplots()
    canvas = FigureCanvasAgg(fig)
    num_samples = pose.shape[0]

    scale = np.ones(num_samples)*15
    c = np.array([mpl.colors.to_hex('Black')]*num_samples)
    
    norm_sim = (similarity-similarity.min())/(similarity-similarity.min()).max()
    
    c1='yellow' #blue
    c2='red' #green
    

    for s,i in zip(norm_sim,map):
        color=colorFader(c1,c2,s)
        c[i] = color
        scale[i] = 50


    c[query] = mpl.colors.to_hex('green')
    scale[query] = 90

    p = ax.scatter(pose[:,0],pose[:,1],s = scale, c = c)
    ax.set_aspect('equal')
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    return X
    

