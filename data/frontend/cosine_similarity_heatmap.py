import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from math import pi
import colorcet as cc
from bokeh.plotting import figure
from bokeh.models import (ColorBar, ColumnDataSource, TextInput, LinearColorMapper, FactorRange, Div, Spinner, 
Dropdown, Switch, Button, BasicTicker, PrintfTickFormatter)

from bokeh.layouts import column, row
from bokeh.palettes import Plasma

    
class CosineSimilarityMatrix:
    def __init__(self, vectors, vector_ids, id_to_cluster_map, vector_id_name):
        
        # Later we convert these to string for bokeh, here we capture the integer versions
        self.ids = vector_ids 

        self.vectors = None
        self.vector_ids = None
        self.vector_id_name = None
        
        self._set_vector_attrs(vectors=vectors, vector_ids=vector_ids, vector_id_name=vector_id_name)
        self.id_col1 = f"{self.vector_id_name} 1"
        self.id_col2 = f"{self.vector_id_name} 2"
        self.id_to_cluster_map = id_to_cluster_map

        self.figure = None 
        self.heatmap = None
        self.figure_title = None 
        
        self.color_mapper = None
        self.color_bar = None
        # self.palette = cc.b_linear_bmw_5_95_c89
        self.palette = cc.linear_blue_5_95_c73
        
        self.similarity_matrix = None 
        self.similarity_df = None 
        self.stacked_similarity_df = None 
        self.source = None 

        self.button_submit_filter = None 

        self.div_filter_sim_matrix = None
        self.switch_filter_sim_matrix = None

        self.div_sim_matrix_filter_type = None
        self.dd_sim_matrix_filter_type = None 

        self.div_sim_matrix_filter_size = None
        self.spinner_sim_matrix_filter_size = None 

        self.div_sim_matrix_focus_id = None
        self.spinner_sim_matrix_focus_id = None 

        self.div_sim_matrix_focus_type = None
        self.dd_sim_matrix_focus_type = None 

        self.div_sim_matrix_filter_exact_ids = None
        self.text_input_sim_matrix_filter_exact_ids = None 

        self.controls = None 

    def _initialize_figure_controls(self, selections):

        self.div_filter_sim_matrix = Div(text="Filter Similarity Matrix:")   
        self.switch_filter_sim_matrix = Switch(active=selections["sim_matrix_filter"])

        # Filter Type
        self.div_sim_matrix_filter_type = Div(text="Filter Type: ", visible=selections["sim_matrix_filter"])
        self.dd_sim_matrix_filter_type = Dropdown(label=f"filter_type={selections['sim_matrix_filter_type']}", 
                                                  menu=[("Random", "random"), 
                                                        ("Focused", "focused"), 
                                                        ("Exact", "exact")], 
                                                  visible=selections["sim_matrix_filter"]) 
                                                  
        # Filter Size
        self.div_sim_matrix_filter_size = Div(text="Filter Size: ", 
                                              visible=selections["sim_matrix_filter"] and\
                                                        selections['sim_matrix_filter_type'] in ['random', 'focused'])
        self.spinner_sim_matrix_filter_size = Spinner(title="", 
                                                      step=1, 
                                                      low=2, 
                                                      high=25,
                                                      value=10, 
                                                      visible=selections["sim_matrix_filter"] and\
                                                        selections['sim_matrix_filter_type'] in ['random', 'focused']) 

        # Focus ID
        self.div_sim_matrix_focus_id = Div(text=f"Focus {self.vector_id_name}: ", 
                                           visible=selections["sim_matrix_filter"] and\
                                                    selections['sim_matrix_filter_type'] == "focused")
        self.spinner_sim_matrix_focus_id = Spinner(title="", 
                                                   step=1, 
                                                   low=np.min(self.ids), 
                                                   high=np.max(self.ids),
                                                   value=1, 
                                                   visible=selections["sim_matrix_filter"] and\
                                                    selections['sim_matrix_filter_type'] == "focused")  

        # Focus Type
        self.div_sim_matrix_focus_type = Div(text="Focus Type: ", 
                                           visible=selections["sim_matrix_filter"] and\
                                                    selections['sim_matrix_filter_type'] == "focused")
        self.dd_sim_matrix_focus_type = Dropdown(label=f"focus_type={selections['sim_matrix_filter_focus_type']}", 
                                                  menu=[("Most Similar", "most_similar"), 
                                                        ("Least Similar", "least_similar"), 
                                                        ("Random", "random")], 
                                                  visible=selections["sim_matrix_filter"] and\
                                                    selections['sim_matrix_filter_type'] == "focused")

        # Exact Filtering
        self.div_sim_matrix_filter_exact_ids = Div(text=f"{self.vector_id_name}'s (comma separated):", 
                                                   visible=selections["sim_matrix_filter"] and\
                                                     selections['sim_matrix_filter_type'] == "exact")
        self.text_input_sim_matrix_filter_exact_ids = TextInput(title="", 
                                                                value='', 
                                                                visible=selections["sim_matrix_filter"] and\
                                                                    selections['sim_matrix_filter_type'] == "exact")      

        self.button_submit_filter = Button(label="Apply Filter", 
                                           button_type="success", 
                                           visible=selections["sim_matrix_filter"])

        self.controls = column(self.div_filter_sim_matrix, self.switch_filter_sim_matrix, 
                               self.div_sim_matrix_filter_type, self.dd_sim_matrix_filter_type, 
                               self.div_sim_matrix_filter_size, self.spinner_sim_matrix_filter_size, 
                               self.div_sim_matrix_focus_id, self.spinner_sim_matrix_focus_id, 
                               self.div_sim_matrix_focus_type, self.dd_sim_matrix_focus_type, 
                               self.div_sim_matrix_filter_exact_ids, self.text_input_sim_matrix_filter_exact_ids, 
                               self.button_submit_filter)    
        
        return 

    def _set_figure_layout(self):
        self.layout = row(self.controls, self.figure)
        return

    def get_figure_layout(self):
        self._set_figure_layout()
        return self.layout
        
    def initialize_figure(self, selections):
        
        self.figure_title = "Heatmap of Cosine Similarities for Vectors Used by Clustering Algorithm"
        
        self._initialize_data_source(selections=selections)

        self._initialize_figure_controls(selections=selections)
        
        x_utilized_ids = self.stacked_similarity_df["V1_ID"].unique().tolist()
        y_utilized_ids = list(reversed(self.stacked_similarity_df["V2_ID"].unique().tolist()))
        
        TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
        self.figure = figure(title=self.figure_title,
                             x_axis_location="above",
                             x_range=FactorRange(factors=x_utilized_ids, name="heatmap_x_range"),
                             y_range=FactorRange(factors=y_utilized_ids, name="heatmap_y_range"),
                             width=700, 
                             height=700,
                             tools=TOOLS, 
                             tooltips=[('Cosine Similarity', '@{cosine_similarity}'), 
                                       (f'{self.id_col1}', f"@{{{self.id_col1}}}"), 
                                       (f'{self.id_col2}', f"@{{{self.id_col2}}}"), 
                                       ("Text1 Cluster", "@{Text1 Cluster}"), 
                                       ("Text2 Cluster", "@{Text2 Cluster}")],
                             toolbar_location='below',
                             )
        
        self.figure.grid.grid_line_color = None
        self.figure.axis.axis_line_color = None
        self.figure.axis.major_tick_line_color = None
        self.figure.axis.major_label_text_font_size = "11px"
        self.figure.axis.major_label_standoff = 0
        self.figure.xaxis.major_label_orientation = pi / 3
        
        # Set color mapper "high" based on the data, rather than metrics allowed range
        f = self.stacked_similarity_df[self.id_col1] != self.stacked_similarity_df[self.id_col2]
        high = self.stacked_similarity_df.loc[f, "cosine_similarity"].max()
        self.color_mapper = LinearColorMapper(palette=self.palette, 
                                              low=-1.0, 
                                              high=1.0)
        
        self.heatmap = self.figure.rect(x="V1_ID", 
                                        y="V2_ID", 
                                        height=1, 
                                        width=1,
                                        source=self.source, 
                                        fill_color={'field': 'cosine_similarity', 
                                                    'transform': self.color_mapper})
       
        self.color_bar = ColorBar(color_mapper=self.color_mapper, 
                                  major_label_text_font_size="10px",
                                  ticker=BasicTicker(desired_num_ticks=10),
                                  display_low=-1.0, 
                                  display_high=1.0,
                                  label_standoff=6, 
                                  border_line_color=None)

        self.figure.add_layout(self.color_bar, 'right')

        return 

    def update_figure(self, selections, vectors=None, vector_ids=None, vector_id_name=None, 
                        id_to_cluster_map=None):

        if vectors is not None and vector_ids is not None and vector_id_name is not None:
            self._set_vector_attrs(vectors=vectors, vector_ids=vector_ids, vector_id_name=vector_id_name)
        
        if id_to_cluster_map:
            self.id_to_cluster_map = id_to_cluster_map

        self._update_data_source(selections=selections)
        
        x_utilized_ids = self.stacked_similarity_df["V1_ID"].unique().tolist()
        y_utilized_ids = list(reversed(self.stacked_similarity_df["V2_ID"].unique().tolist()))

        self.figure.select({'name':'heatmap_x_range'}).update(factors=x_utilized_ids)
        self.figure.select({'name':'heatmap_y_range'}).update(factors=y_utilized_ids)
        
        return
    
    def _initialize_data_source(self, selections):
        
        self._setup_data_source(selections=selections)
        
        self.source = ColumnDataSource(self.stacked_similarity_df)
        
        return 
        
    def _update_data_source(self, selections):
        
        self._setup_data_source(selections=selections)
        
        self.source.data = ColumnDataSource.from_df(self.stacked_similarity_df)
        
        return 

    def _setup_data_source(self, selections=None):
        
        self._calculate_similarity()
        
        self._apply_filters(selections=selections)
        
        return
    
    def _apply_filters(self, selections):
        
        if selections and selections['sim_matrix_filter']:
            
            if selections['sim_matrix_filter_type'] == "random":
                self._apply_random_filter(filter_size=selections['sim_matrix_filter_size'])
                
            elif selections['sim_matrix_filter_type'] == "focused":
                self._apply_focused_filter(focus_id=selections['sim_matrix_filter_focus_id'], 
                                           focus_type=selections['sim_matrix_filter_focus_type'], 
                                           filter_size=selections['sim_matrix_filter_size'])
            elif selections['sim_matrix_filter_type'] == "exact":
                self._apply_exact_filter(exact_ids=selections['sim_matrix_filter_exact_ids'])
        
        return 
        
    def _apply_exact_filter(self, exact_ids):
        
        f1 = self.stacked_similarity_df[self.id_col1].isin(exact_ids)
        f2 = self.stacked_similarity_df[self.id_col2].isin(exact_ids)
        f = f1 & f2
        self.stacked_similarity_df = self.stacked_similarity_df.loc[f, :]
        return
    
    def _apply_focused_filter(self, focus_id, focus_type, filter_size):
        
        if focus_type == "most_similar":
            
            focus_filter1 = self.stacked_similarity_df[self.id_col1] == focus_id
            focus_filter2 = self.stacked_similarity_df[self.id_col2] != focus_id
            focus_filter = focus_filter1 & focus_filter2
            focus_df = self.stacked_similarity_df.loc[focus_filter, :]
            focus_df = focus_df.sort_values(by="cosine_similarity", ascending=False)
            
            other_ids = focus_df[self.id_col2].tolist()[:filter_size-1]
            filtered_ids = [focus_id] + other_ids
            
            f1 = self.stacked_similarity_df[self.id_col1].isin(filtered_ids)
            f2 = self.stacked_similarity_df[self.id_col2].isin(filtered_ids)
            f = f1 & f2
            
            self.stacked_similarity_df = self.stacked_similarity_df.loc[f, :].copy(deep=True) 
            
        elif focus_type == "least_similar":
            
            focus_filter1 = self.stacked_similarity_df[self.id_col1] == focus_id
            focus_filter2 = self.stacked_similarity_df[self.id_col2] != focus_id
            focus_filter = focus_filter1 & focus_filter2
            focus_df = self.stacked_similarity_df.loc[focus_filter, :]
            focus_df = focus_df.sort_values(by="cosine_similarity", ascending=True)
            
            other_ids = focus_df[self.id_col2].tolist()[:filter_size-1]
            filtered_ids = [focus_id] + other_ids
            
            f1 = self.stacked_similarity_df[self.id_col1].isin(filtered_ids)
            f2 = self.stacked_similarity_df[self.id_col2].isin(filtered_ids)
            f = f1 & f2
            
            self.stacked_similarity_df = self.stacked_similarity_df.loc[f, :].copy(deep=True) 
            
        elif focus_type == "random":
            
            focus_filter1 = self.stacked_similarity_df[self.id_col1] == focus_id
            focus_filter2 = self.stacked_similarity_df[self.id_col2] != focus_id
            focus_filter = focus_filter1 & focus_filter2
            focus_df = self.stacked_similarity_df.loc[focus_filter, :]
            
            other_ids = np.random.choice(a=focus_df[self.id_col2].tolist(), size=filter_size-1)
            filtered_ids = [focus_id] + list(other_ids)
            
            f1 = self.stacked_similarity_df[self.id_col1].isin(filtered_ids)
            f2 = self.stacked_similarity_df[self.id_col2].isin(filtered_ids)
            f = f1 & f2
            
            self.stacked_similarity_df = self.stacked_similarity_df.loc[f, :].copy(deep=True) 
        
        return
    
    def _apply_random_filter(self, filter_size):
        
        random_ids = np.random.choice(a=self.vector_ids, size=filter_size)
        
        f1 = self.stacked_similarity_df[self.id_col1].isin(random_ids)
        f2 = self.stacked_similarity_df[self.id_col2].isin(random_ids)
        f = f1 & f2
        self.stacked_similarity_df = self.stacked_similarity_df.loc[f, :]
        
        return 
    
    def _calculate_similarity(self):
         
        self.similarity_matrix = cosine_similarity(X=self.vectors)
        
        self.similarity_df = pd.DataFrame(self.similarity_matrix, 
                                          columns=self.vector_ids, 
                                          index=self.vector_ids)
        
        self.stacked_similarity_df = pd.DataFrame(self.similarity_df.stack(), 
                                                  columns=["cosine_similarity"])\
                                        .reset_index()\
                                        .rename(columns={"level_0":self.id_col1, 
                                                         "level_1":self.id_col2})
        
        
        self.stacked_similarity_df["V1_ID"] = (f"{self.id_col1}: " + self.stacked_similarity_df[self.id_col1].to_numpy()).tolist()
        self.stacked_similarity_df["V2_ID"] = (f"{self.id_col2}: " + self.stacked_similarity_df[self.id_col2].to_numpy()).tolist()

        self.stacked_similarity_df['Text1 Cluster'] = [self.id_to_cluster_map[txt_id] for txt_id in self.stacked_similarity_df[self.id_col1].tolist()]
        self.stacked_similarity_df['Text2 Cluster'] = [self.id_to_cluster_map[txt_id] for txt_id in self.stacked_similarity_df[self.id_col2].tolist()]
        
        return
    
    def _set_vector_attrs(self, vectors, vector_ids, vector_id_name):
        self.vectors = vectors
        
        # Used to label axes for the matrix
        self.vector_ids = [str(vector_id) for vector_id in vector_ids] 
        self.vector_id_name = vector_id_name
        return 