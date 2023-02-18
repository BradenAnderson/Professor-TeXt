
from settings import *
from data.backend.text_viz_generator import TextVizGenerator

#from functools import partial 

import numpy as np
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Dropdown, HoverTool, RadioGroup, RadioButtonGroup
from bokeh.layouts import column, row
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma
from bokeh.events import ButtonClick, MenuItemClick

from data.frontend.interactive_dashboard import BokehDashboard

# Instantiate Dashboard
# bokeh_doc = curdoc()
# bokeh_doc.title = "MIST"

app = BokehDashboard(selections=current_selections, paths=paths)

# app.setup_load_screen()
# bokeh_doc.add_root(app.dashboard.file_upload_tab)

app.setup()

# bokeh_doc.add_root(app.dashboard.tabs)
