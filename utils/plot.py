

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os 
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio

clear = lambda: os.system('cls')


def instance_plot(x,y,s=20,c='b'):
    fig = Figure(figsize=(5, 4), dpi=100)
    fig, ax = plt.subplots()
    ax.scatter(x,y,s = s, c = c)
    ax.set_aspect('equal')
    #fig.canvas.draw()



class myplot():
    def __init__(self,delay = 0.01,offset=20):
        self.fig = Figure(figsize=(5, 4), dpi=100,)
        self.fig, self.ax = plt.subplots()
        self.gif_record_flag = False
        self.delay = delay
        self.offset = offset

    def record_gif(self, filename = 'path.gif'):
        self.gif_record_flag = True
        self.canvas = FigureCanvasAgg(self.fig)
        self.writer = imageio.get_writer(filename, mode='I')

    def init_plot(self,x,y,s,c):
        self.p = self.ax.scatter(x,y,s = s, c = c)
    
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

        self.ax.axis([xmin - offset, xmax + offset, ymin -offset, ymax + offset])
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

    
def init_plot(x,y,s,c):
    p = ax.scatter(x,y,s = 20, c = 'g')
    return(p)

def update_plot(p,x,y,offset=20,color=[],zoom=10):
    global ax 
    p.set_offsets(np.c_[x,y])

    if color != []:
        p.set_color(color)

    if len(x)<zoom:
        xmax,xmin= max(x),min(x)
        ymax,ymin = max(y),min(y)
    else: 
        xmax,xmin= max(x[-zoom:]),min(x[-zoom:])
        ymax,ymin = max(y[-zoom:]),min(y[-zoom:])

    ax.axis([xmin - offset, xmax + offset, ymin -offset, ymax + offset])
    ax.set_aspect('equal')
    fig.canvas.draw()
    plt.pause(0.001)
    return(p)


def dynamic_plot(x,y,step=10,s = 20, c = 'g'):
    p = init_plot(x[0],y[0],20,'b')
    leng = len(x)
    for i in range(0,leng,step):
        xi,yi = x[:i+1],y[:i+1]
        update_plot(p,xi,yi,offset=20,zoom=0)
