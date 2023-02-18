import numpy as np
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import (ColumnDataSource, Slider, Dropdown, HoverTool, TextInput, 
                Range1d, WheelZoomTool, ResetTool, PanTool, SaveTool, RadioButtonGroup, 
                Div, Spinner, Switch)
from bokeh.layouts import column, row
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma

class EmbeddingScatterPlot:
    def __init__(self, text_viz_df, color_settings, selections):
        
        self.text_viz_df = text_viz_df
        self.source = ColumnDataSource(self.text_viz_df)

        self.hover_columns = selections['embedding_scatter_hover_columns']
        self.tool_tips = self._get_tooltips(selections=selections)
        self.axis_range_buffer_pct = selections["axis_range_buffer_pct"]

        self.hover = HoverTool(tooltips=self.tool_tips, attachment="below")

        #self.box_zoom = BoxZoomTool()
        #self.full_screen = FullscreenTool()

        self.figure = None
        self.figure_title = None
        self.scatter = None
        self.scatter_size = 20
        self.color_map = linear_cmap(field_name="cluster", 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])
        # self.selections = None

        # Embedding controls
        self.dropdown_embedding = None
        self.dd_hugging_model_checkpoint = None
        self.dd_hugging_embedding_type = None # Mean, start sequence token only, etc.
        self.text_hugging_model_checkpoint = None 
        ## TFIDF Hyperparams
        self.dd_tfidf_strip_accents = None
        self.dd_tfidf_norm = None
        
        self.div_tfidf_max_df = None
        self.radio_tfidf_max_df = None
        self.slider_tfidf_max_df = None
        self.spinner_tfidf_max_df = None

        self.div_tfidf_min_df = None
        self.radio_tfidf_min_df = None
        self.slider_tfidf_min_df = None
        self.spinner_tfidf_min_df = None

        self.switch_tfidf_lowercase = None
        self.switch_tfidf_binary = None
        self.switch_tfidf_use_idf = None
        self.switch_tfidf_smooth_idf = None
        self.switch_tfidf_sublinear_tf = None 

        self.div_tfidf_lowercase = None
        self.div_tfidf_binary = None
        self.div_tfidf_use_idf = None
        self.div_tfidf_smooth_idf = None
        self.div_tfidf_sublinear_tf = None

        # Dimensionality reduction controls
        self.dropdown_dimreduce = None
        self.dd_scale_before_dim_reduce = None
        self.dd_umap_metric = None
        self.slider_tsne_perplexity = None
        self.slider_umap_n_neighbors = None

        # Cluster controls
        self.dropdown_cluster = None
        self.dd_cluster_space = None
        self.dd_scale_before_cluster = None
        self.slider_n_clusters = None 
        self.dd_agg_affinity = None 
        self.dd_agg_linkage = None 
        self.slider_kmeans_ninit = None
        self.dd_kmeans_algo = None

        self.controls_layout = None
        self.embedding_controls = None
        self.dim_reduce_controls = None
        self.cluster_controls = None

        return 
    
    def _get_tooltips(self, selections):

        TOOLTIPS = '<div>'
        
        for column in self.hover_columns:
            col = (
                f'<div style="width:{selections["hover_window_width"]};">'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">{column}: </span>' 
                f'<span style="font-size: {selections["hover_text_size"]};">@{{{column}}}</span><br>'
                '</div>'
                   )
            TOOLTIPS += col
        
        TOOLTIPS += ('<br>'
                        '</div>')

        return TOOLTIPS

    def get_controls_layout(self):
        return self.controls_layout 
        
    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)

        self.figure = figure(title=self.figure_title, 
                             tools=[self.hover, WheelZoomTool(), ResetTool(), PanTool(), SaveTool()], 
                             width=selections["embedding_scatter_chart_width"], 
                             height=selections["embedding_scatter_chart_height"])
        
        self.scatter = self.figure.circle(x="x_embed", 
                                          y="y_embed", 
                                          name="embedding_scatter_chart",
                                          size=self.scatter_size, 
                                          fill_color=self.color_map, 
                                          alpha=0.5, 
                                          source=self.source, 
                                          visible=True)

        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.x_range = Range1d(start=x_min, 
                                      end=x_max, 
                                      name="embed_scatter_xrange")

        self.figure.y_range = Range1d(start=y_min, 
                                      end=y_max, 
                                      name="embed_scatter_yrange")

        self.initialize_figure_controls(selections=selections)

        return 

    def _get_x_y_axis_ranges(self):
        x_range = self.text_viz_df['x_embed'].max() - self.text_viz_df['x_embed'].min()
        y_range = self.text_viz_df['y_embed'].max() - self.text_viz_df['y_embed'].min()
        x_min = self.text_viz_df['x_embed'].min() - x_range*self.axis_range_buffer_pct
        x_max = self.text_viz_df['x_embed'].max() + x_range*self.axis_range_buffer_pct
        y_min = self.text_viz_df['y_embed'].min() - y_range*self.axis_range_buffer_pct
        y_max = self.text_viz_df['y_embed'].max() + y_range*self.axis_range_buffer_pct
        return x_min, x_max, y_min, y_max

    def initialize_figure_controls(self, selections):

        self._initialize_embedding_controls(selections=selections)
        self._initialize_dimensionality_reduction_controls(selections=selections)
        self._initialize_cluster_controls(selections=selections)

        self.controls_layout = row(self.embedding_controls, 
                                   self.dim_reduce_controls, 
                                   self.cluster_controls)

        return 

    def _initialize_embedding_controls(self, selections):

        self.dropdown_embedding = Dropdown(label=f"Embedding Type = {selections['embedding_type']}", 
                                           name="select_embedding", 
                                           button_type="success",
                                           menu=[("TFIDF", "tfidf"),  
                                                 ("FastText Mean", "fasttext_mean"), 
                                                 ("FastText TFIDF", "fasttext_tfidf"), 
                                                 ("OpenAI Ada Model", "openai_ada__text-embedding-ada-002"), 
                                                 ("Hugging Face Transformer", "hugging_face")])

        self.dd_hugging_model_checkpoint = Dropdown(label=f"HuggingFace Model={selections['hugging_face_model_checkpoint']}", 
                                                    name="select_hugging_model_checkpoint", 
                                                    menu=[("distilbert-base-uncased", "distilbert-base-uncased"), 
                                                          ("bert-base-uncased", "bert-base-uncased"), 
                                                          ("bert-large-uncased", "bert-large-uncased"), 
                                                          ("gpt2", "gpt2"), 
                                                          ("gpt2-large", "gpt2-large"), 
                                                          ("gpt2-xl", "gpt2-xl"), 
                                                          ("roberta-base", "roberta-base"), 
                                                          ("roberta-large","roberta-large"), 
                                                          ("reformer-enwik8", "reformer-enwik8"), 
                                                          ("albert-base-v2", "albert-base-v2"), 
                                                          ("albert-large-v2", "albert-large-v2"), 
                                                          ("albert-xlarge-v2", "albert-xlarge-v2"), 
                                                          ("albert-xxlarge-v2", "albert-xxlarge-v2")], 
                                                    visible=selections['embedding_type']=="hugging_face")

        t = "Optional: custom hugging face checkpoint"
        self.text_hugging_model_checkpoint = TextInput(name="user_input_hugging_face_model", 
                                                       title=t,
                                                       value="", 
                                                       visible=selections['embedding_type']=="hugging_face")

        self.dd_hugging_embedding_type = Dropdown(label=f"Hidden State Embedding={selections['hugging_face_embedding_type']}", 
                                                  name="select_hugging_embedding_type", 
                                                  menu=[("Start Sequence Token Hidden State Only", "start_token_hidden"), 
                                                        ("Average Hidden States for All Tokens", "mean")], 
                                                  visible=selections['embedding_type']=="hugging_face")


        ## TFIDF
        # tfidf strip_accents
        self.dd_tfidf_strip_accents = Dropdown(label=f"strip_accents={selections['tfidf_strip_accents']}", 
                                               name="tfidf_strip_accents", 
                                               menu=[("None", "None"), 
                                                     ("ascii", "ascii"), 
                                                     ("unicode", "unicode")],
                                               visible=selections['embedding_type']=="tfidf")
        # tfidf norm
        self.dd_tfidf_norm = Dropdown(label=f"norm={selections['tfidf_norm']}", 
                                      name="tfidf_norm", 
                                      menu=[("L2", "l2"), 
                                            ("L1", "l1"), 
                                            ("None", "None")],
                                      visible=selections['embedding_type']=="tfidf")
        
        # tfidf max_df
        self.div_tfidf_max_df = Div(text="<b>max_df as document:</b> ", 
                                    name="tfidf_max_df_div",
                                    visible=selections['embedding_type']=="tfidf")
        self.radio_tfidf_max_df = RadioButtonGroup(labels=["proportion", "count"],  
                                                   name="tfidf_max_df_radio",
                                                   active=0, 
                                                   visible=selections['embedding_type']=="tfidf")
        self.slider_tfidf_max_df = Slider(title="max_df document proportion",   
                                          name="tfidf_max_df_slider",
                                          start=0.0, 
                                          end=1.0, 
                                          step=0.0001, 
                                          format='0[.]0000',
                                          value=selections['tfidf_max_df_proportion'], 
                                          visible=selections['embedding_type']=="tfidf" and selections['tfidf_specify_max_df_as']=="proportion")
        self.spinner_tfidf_max_df = Spinner(title="max_df document count", 
                                            name="tfidf_max_df_spinner",
                                            step=1, 
                                            low=0,
                                            value=selections['tfidf_max_df_count'], 
                                            visible=selections['embedding_type']=="tfidf" and selections['tfidf_specify_max_df_as']=="count")

        # tfidf min_df
        self.div_tfidf_min_df =  Div(text="<b>min_df as document:</b> ", 
                                     visible=selections['embedding_type']=="tfidf")
        self.radio_tfidf_min_df = RadioButtonGroup(labels=["proportion", "count"],  
                                                   name="tfidf_min_df_radio",
                                                   active=1, 
                                                   visible=selections['embedding_type']=="tfidf")
        self.slider_tfidf_min_df = Slider(title="min_df document proportion", 
                                          name="tfidf_min_df_slider",
                                          start=0.0, 
                                          end=1.0, 
                                          step=0.0001, 
                                          format='0[.]0000',
                                          value=selections['tfidf_min_df_proportion'], 
                                          visible=selections['embedding_type']=="tfidf" and selections['tfidf_specify_min_df_as']=="proportion")
        self.spinner_tfidf_min_df = Spinner(title="min_df document count", 
                                            name="tfidf_min_df_spinner",
                                            step=1, 
                                            low=0,
                                            value=selections['tfidf_min_df_count'],
                                            visible=selections['embedding_type']=="tfidf" and selections['tfidf_specify_min_df_as']=="count")

        self.div_tfidf_lowercase = Div(text="<b>Lowercase:</b> ", 
                                       name="tfidf_div_lowercase",
                                       visible=selections['embedding_type']=="tfidf")
        self.switch_tfidf_lowercase = Switch(active=True, visible=selections['embedding_type']=="tfidf")

        self.div_tfidf_binary = Div(text="<b>Binary:</b> ", 
                                    name="tfidf_div_binary",
                                    visible=selections['embedding_type']=="tfidf")
        self.switch_tfidf_binary = Switch(active=False, visible=selections['embedding_type']=="tfidf")

        self.div_tfidf_use_idf = Div(text="<b>use_idf:</b> ", 
                                     name="tfidf_div_use_idf",
                                     visible=selections['embedding_type']=="tfidf")
        self.switch_tfidf_use_idf = Switch(active=True, visible=selections['embedding_type']=="tfidf")

        self.div_tfidf_smooth_idf = Div(text="<b>smooth_idf:</b> ", 
                                        name="tfidf_div_smooth_idf",
                                        visible=selections['embedding_type']=="tfidf")
        self.switch_tfidf_smooth_idf = Switch(active=True, visible=selections['embedding_type']=="tfidf")

        self.div_tfidf_sublinear_tf = Div(text="<b>sublinear_tf:</b> ", 
                                          name="tfidf_div_sublinear_tf",
                                          visible=selections['embedding_type']=="tfidf")
        self.switch_tfidf_sublinear_tf = Switch(active=False, visible=selections['embedding_type']=="tfidf") 


        self.embedding_controls = column(self.dropdown_embedding, self.dd_hugging_model_checkpoint, 
                                         self.text_hugging_model_checkpoint, self.dd_hugging_embedding_type, 
                                         self.dd_tfidf_strip_accents, self.dd_tfidf_norm, 
                                         row(self.div_tfidf_max_df, self.radio_tfidf_max_df), 
                                         self.slider_tfidf_max_df, self.spinner_tfidf_max_df, 
                                         row(self.div_tfidf_min_df, self.radio_tfidf_min_df), 
                                         self.slider_tfidf_min_df, self.spinner_tfidf_min_df, 
                                         row(self.div_tfidf_lowercase, self.switch_tfidf_lowercase), 
                                         row(self.div_tfidf_binary, self.switch_tfidf_binary), 
                                         row(self.div_tfidf_use_idf, self.switch_tfidf_use_idf), 
                                         row(self.div_tfidf_smooth_idf, self.switch_tfidf_smooth_idf), 
                                         row(self.div_tfidf_sublinear_tf, self.switch_tfidf_sublinear_tf))

        return 

    def _initialize_dimensionality_reduction_controls(self, selections):

        self.dropdown_dimreduce = Dropdown(label=f"Dimensionality Reduction = {selections['dimreduce_type']}", 
                                           name="select_dim_reduction", 
                                           background="#000000",
                                           menu=[("UMAP", "umap"), 
                                                 ("Linear","linear"), 
                                                 ("TSNE","tsne"), 
                                                 ("Linear --> TSNE", "linear__tsne")])
        
        self.dd_umap_metric = Dropdown(label=f"UMAP Metric = {selections['umap_metric']}", 
                                       name="select_umap_metric", 
                                       menu=["euclidean", 
                                             "manhattan", 
                                             "chebyshev", 
                                             "hellinger"], 
                                       visible=False)

        self.dd_scale_before_dim_reduce = Dropdown(label=f"Scaling Before Dim Reduce = {selections['scale_before_dim_reduce']}", 
                                                   name="select_scaling_method", 
                                                   menu=[("None", "None"), 
                                                         ("MinMax (zero to one range)", "min_max"), 
                                                         ("Zero Mean, Unit Variance", "standard_scaler")], 
                                                   visible=True)
        
        self.slider_step1_dimreduce_ndims = Slider(title="Dim Reduce Step 1, Output Dimensionality", 
                                                   start=10, 
                                                   end=100, 
                                                   value=selections['step1_dimreduce_ndims'], 
                                                   visible=len(selections['dimreduce_type'].split("__")) > 1)

        self.slider_tsne_perplexity = Slider(title="TSNE Perplexity", 
                                             start=5, 
                                             end=100, 
                                             value=selections['tsne_perplexity'], 
                                             visible="tsne" in selections["dimreduce_type"])

        self.slider_umap_n_neighbors = Slider(title=f"Number of Neighbors",
                                              start=2, 
                                              end=100, 
                                              value=selections['umap_n_neighbors'],
                                              visible=False)

        self.dim_reduce_controls = column(self.dropdown_dimreduce, 
                                          self.dd_scale_before_dim_reduce, 
                                          self.slider_step1_dimreduce_ndims,
                                          self.dd_umap_metric, 
                                          self.slider_tsne_perplexity, 
                                          self.slider_umap_n_neighbors)
        return

    def _initialize_cluster_controls(self, selections):

        self.dropdown_cluster = Dropdown(label=f"Cluster Algorithm = {selections['cluster']}", 
                                         name="select_cluster", 
                                         menu=[("Agglomerative","agglomerative"), 
                                               ("KMeans","kmeans")])
        
        self.dd_cluster_space = Dropdown(label=f"Cluster Space = {selections['cluster_space']}", 
                                        name="select_when_to_cluster", 
                                        menu=[("Auto","auto"), 
                                              ("Before Dim Reduce","before_dim_reduce"), 
                                              ("After Dim Reduce Step1", "after_dim_reduce_step1"),
                                              ("After Full Dim Reduce", "after_full_dim_reduce")], 
                                        visible=True)

        self.dd_scale_before_cluster = Dropdown(label=f"Scaling Before Clustering = {selections['scale_before_cluster']}", 
                                                   name="select_scaling_method_before_cluster", 
                                                   menu=[("None", "None"), 
                                                         ("MinMax (zero to one range)", "min_max"), 
                                                         ("Zero Mean, Unit Variance", "standard_scaler")], 
                                                   visible=True)

        self.slider_n_clusters = Slider(title="Number of Clusters",
                                        start=2, 
                                        end=20, 
                                        value=selections['n_clusters'],
                                        visible=True)

        self.dd_agg_affinity = Dropdown(label=f"affinity = {selections['agg_cluster_affinity']}", 
                                        name="select_agg_affinity", 
                                        menu=[("Cosine","cosine"), 
                                              ("Euclidean","euclidean"), 
                                              ("Manhattan", "manhattan"),
                                              ("L1", "l1"), 
                                              ("L2", "l2")], 
                                        visible=selections["cluster"]=="agglomerative")

        self.dd_agg_linkage = Dropdown(label=f"linkage = {selections['agg_cluster_linkage']}", 
                                       name="select_agg_linkage", 
                                       menu=[("Average","average"), 
                                             ("Complete","complete"), 
                                             ("Ward", "ward"),
                                             ("Single", "single")], 
                                       visible=selections["cluster"]=="agglomerative")
        
        self.slider_kmeans_ninit = Slider(title="Number of Initializations",
                                          start=5, 
                                          end=200, 
                                          value=selections['kmeans_n_init'],
                                          visible=selections['cluster']=="kmeans")
        
        self.dd_kmeans_algo = Dropdown(label=f"algorithm = {selections['kmeans_algorithm']}", 
                                       name="select_kmeans_algo", 
                                       menu=[("Lloyd","lloyd"), 
                                             ("Elkan", "elkan"),
                                             ("Auto", "auto")], 
                                             visible=selections["cluster"]=="kmeans")

        self.cluster_controls = column(self.dropdown_cluster, 
                                       self.dd_scale_before_cluster,
                                       self.dd_cluster_space,
                                       self.slider_n_clusters, 
                                       self.dd_agg_affinity, 
                                       self.dd_agg_linkage, 
                                       self.slider_kmeans_ninit, 
                                       self.dd_kmeans_algo)

        return 


    def update_figure(self, df, selections, color_settings):

        self.figure.title.text = self.get_figure_title(selections=selections)
        
        self.color_map = linear_cmap(field_name="cluster", 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])

        self.figure.select({'name':'embedding_scatter_chart'}).glyph.update(fill_color=self.color_map)
        
        self.update_data_source(df=df)
        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.select({'name':'embed_scatter_xrange'}).update(start=x_min, end=x_max)
        self.figure.select({'name':'embed_scatter_yrange'}).update(start=y_min, end=y_max)

        return 

    def update_data_source(self, df):
        self.text_viz_df = df
        self.source.data = ColumnDataSource.from_df(df)
        return 

    def get_figure_title(self, selections):

        if selections["embedding_type"] != "hugging_face":
            embed_str = f"Embedding={selections['embedding_type']}"
        elif selections['embedding_type'] == 'hugging_face': 
            embed_str = f"Embedding=Hugging Face {selections['hugging_face_model_checkpoint']} ({selections['hugging_face_embedding_type']})"

        reduce_str = f"Reduction={selections['dimreduce_type']}"

        cluster_str = f"Cluster={selections['cluster']} "
        if selections['cluster'] == "agglomerative":
            cluster_str += f"(Num Clusters={selections['n_clusters']}, affinity={selections['agg_cluster_affinity']})"
        elif selections['cluster'] == "kmeans":
            cluster_str += f"(Num Clusters={selections['n_clusters']}, algorithm={selections['kmeans_algorithm']})"


        title = (f"{embed_str}, {reduce_str}, {cluster_str}\n"
                 f"{self.text_viz_df.name}")

        return title