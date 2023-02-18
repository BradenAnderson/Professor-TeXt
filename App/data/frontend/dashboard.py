import numpy as np
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import Tabs
from bokeh.layouts import column, row
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma

from .file_upload_tab import FileUploadTab
from .nlp_tab import NLPTab
from .regression_tab import RegressionTab
from .text_generation_tab import TextGenerationTab

class Dashboard:
    def __init__(self, selections, paths, df=None):
        
        self.selections = selections
        self.paths = paths

        self.text_column = selections['text_column']
        self.primary_key_column = None
        self.df = None # setup in setup_data or interactive dashboard file_upload_callback

        self.nlp_tab = None
        self.regression_tab = None
        self.text_generation_tab = None
        self.file_upload_tab = None 

        self.tabs = None  # Full layout is a set of bokeh.models.Tabs

        return

    def update_selections(self, **kwargs):
        
        for key, value in kwargs.items():
            self.selections[key] = value
        
        self._sync_tab_selections()

        return
        
    def _sync_tab_selections(self):

        if self.nlp_tab:
            self.nlp_tab.set_selections(selections=self.selections)
        
        if self.regression_tab:
            self.regression_tab.set_selections(selections=self.selections)
        
        if self.text_generation_tab:
            self.text_generation_tab.set_selections(selections=self.selections)

        self.file_upload_tab.set_selections(selections=self.selections)

        return 

    def setup_dashboard(self):

        self.setup_data()

        print("Setting up NLP Tab")
        self._setup_nlp_tab()

        print("Setting up Regression tab")
        self._setup_regression_tab()

        print("Setting up text generation tab")
        self._setup_text_generation_tab()

        print("Setting up layout")
        self._setup_layout()

        return 

    def update_tabs(self):

        self.update_nlp_tab()

        self.update_regression_tab()

        return

    def update_nlp_tab(self):

        self.nlp_tab.set_selections(selections=self.selections)

        self.nlp_tab.update_plots()

        return 

    def update_regression_tab(self):

        self.regression_tab.set_selections(selections=self.selections)

        self.regression_tab.set_text_viz_df(text_viz_df=self.nlp_tab.text_viz_df)

        self.regression_tab.update_plots()

        return 

    def update_text_generation_tab(self):

        self.text_generation_tab.set_selections(selections=self.selections)

        self.text_generation_tab.set_text_viz_df(text_viz_df=self.nlp_tab.text_viz_df)

        self.text_generation_tab.update()

        # TODO: set this to True in any of the callbacks that would require that we hit the twitter api again.
        #       text_gen_twitter_username, text_gen_tweets_to_exclude, text_gen_num_recent_tweet_examples
        self.selections['text_gen_run_tweet_retriever'] = False

        return 

    def setup_file_upload_tab(self):

        self.file_upload_tab = FileUploadTab()

        self.file_upload_tab.initialize_controls()

        self.file_upload_tab.set_selections(selections=self.selections)

        self.file_upload_panel = self.file_upload_tab.get_layout()

        self.tabs = Tabs(tabs=[self.file_upload_panel], tabs_location="above")

        return

    def _setup_nlp_tab(self):

        self.nlp_tab = NLPTab(df=self.df, 
                              paths=self.paths, 
                              text_column=self.text_column, 
                              primary_key_column=self.primary_key_column)

        self.nlp_tab.set_selections(selections=self.selections)

        self.nlp_tab.setup_plots()

        self.selections['current_cluster_cmap_info'] = self.nlp_tab.current_colors

        return 
    
    def _setup_regression_tab(self):
        
        self.regression_tab = RegressionTab(primary_key_column=self.primary_key_column)

        self.regression_tab.set_selections(selections=self.selections)

        self.regression_tab.set_text_viz_df(text_viz_df=self.nlp_tab.text_viz_df)

        self.regression_tab.setup_plots()

        return 

    def _setup_text_generation_tab(self):
        
        self.text_generation_tab = TextGenerationTab()

        self.text_generation_tab.set_selections(selections=self.selections)

        self.text_generation_tab.set_text_viz_df(text_viz_df=self.nlp_tab.text_viz_df)

        self.text_generation_tab.setup()

        return 

    def _setup_layout(self):

        self.nlp_panel = self.nlp_tab.get_layout()

        self.regression_panel = self.regression_tab.get_layout()

        self.text_generation_panel = self.text_generation_tab.get_layout()

        #self.tabs = Tabs(tabs=[self.nlp_panel, self.regression_panel, self.text_generation_panel], tabs_location="above")

        return

    def get_layout(self):

        return self.tabs 

    def setup_data(self):
        
        if self.selections['run_with_sample_data']:
            print("Loading sample data...")
            self.df = pd.read_csv(self.paths['data_path'])

        if self.selections['primary_key_column'] not in self.df.columns:
            self.df[self.selections['primary_key_column']] = [num for num in range(1, self.df.shape[0] + 1)]

        self.text_column = self.selections['text_column']
        self.df['Txt'] = self.df[self.text_column].to_numpy()
        self.primary_key_column = self.selections['primary_key_column']

        return 

    def set_dataframe(self, df):
        self.df = df
        return