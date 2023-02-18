import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d, WheelZoomTool, ResetTool, PanTool, SaveTool
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import  Plasma

from PIL import Image
import base64
import io

class TextPipelinePlot:
    def __init__(self):
        
        self.graph = None

        self.pillow_image = None
        self.xdim = None 
        self.ydim = None 
        self.numpy_image = None 
        self.view = None 

        self.figure = None 
    
    def _graphviz_to_array(self, graph):
        
        self.graph = graph

        self.pillow_image = Image.open(io.BytesIO(self.graph.draw(format="png"))).convert("RGBA")

        self.xdim, self.ydim = self.pillow_image.size

        self.numpy_image = np.empty((self.ydim, self.xdim), dtype=np.uint32)

        self.view = self.numpy_image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))

        self.view[:,:,:] = np.flipud(np.asarray(self.pillow_image))

        return 
                             
    def initialize_figure(self, graph):
        
        self._graphviz_to_array(graph=graph)

        self.figure = figure(title="Text Visualization Pipeline", 
                             width=self.xdim, 
                             height=self.ydim,
                             x_range=(0, self.xdim), 
                             y_range=(0, self.ydim))

        self.figure.grid.grid_line_color = None

        self.figure.image_rgba(image=[self.numpy_image], 
                               x=0, 
                               y=0, 
                               dw=self.xdim,
                               dh=self.ydim, 
                               name="pipeline_image")
        
        return 

    def update_figure(self, graph):
        
        self._graphviz_to_array(graph=graph)

        self.figure.select({'name':'pipeline_image'}).data_source.data = {'image': [self.numpy_image]}

        return 
