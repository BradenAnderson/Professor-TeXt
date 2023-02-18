import numpy as np
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Dropdown, HoverTool, RadioGroup, RadioButtonGroup, SaveTool
from bokeh.layouts import column, row
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma

class MetricClusterBarPlot:
    def __init__(self, text_viz_df, selections, color_settings):
        
        self.text_viz_df = text_viz_df
        self.metric_df = None
        self.source = None

        self.figure = None
        self.figure_title = None
        self.bar = None
        self.color_map = linear_cmap(field_name="cluster", 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])

        # NOTE: have to keep 'Spend' first or update current_selections in settings.py
        self.column_labels = selections['numeric_metric_columns']
        self.cluster_bar_column_radio = RadioButtonGroup(labels=self.column_labels,  active=0)

        self.cluster_metrics = ["mean", "median", "min", "max", "sum", "std", "var", "sem", "size"]
        self.cluster_bar_metric_radio = RadioButtonGroup(labels=self.cluster_metrics,  active=0)

        self.tool_tips = [("Cluster Number", "@{cluster}"), 
                          ("Cluster Size", "@{size}"), 
                          ("Min", "@{min}"), 
                          ("Mean", "@{mean}"),
                          ("Median", "@{median}"), 
                          ("Max", "@{max}"), 
                          ("Standard Deviation", "@{std}"), 
                          ("Variance", "@{var}"), 
                          ("Standard Error of Mean", "@{sem}")] 

        self.hover = HoverTool(tooltips=self.tool_tips)  

    def update_figure(self, df, selections, color_settings=None):

        self.figure.title.text = self.get_figure_title(selections=selections)

        if color_settings:
            self.color_map = linear_cmap(field_name="cluster", 
                                         palette=color_settings['palette'], 
                                         low=color_settings['low'], 
                                         high=color_settings['high'])

            self.figure.select({'name':'cluster_bar_chart'}).glyph.update(fill_color=self.color_map)
        
        self.update_data_source(selections=selections, df=df)

        return 

    def update_data_source(self, selections, df):

        if df is not None:
            self.text_viz_df = df
        
        self.create_metric_df(selections=selections)

        self.source.data = ColumnDataSource.from_df(self.metric_df)

        return 

    def create_data_source(self, selections):

        self.create_metric_df(selections=selections)

        self.source = ColumnDataSource(self.metric_df)

        return 

    def get_figure_title(self, selections):
        if selections['cluster_bar_metric_plotted'] != "size":
            return f"{selections['cluster_bar_metric_plotted'].title()} {selections['cluster_bar_column_plotted']} by cluster"
        else:
            return f"Number of Observations in Each Cluster"

    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)

        self.create_data_source(selections=selections)

        self.figure = figure(title=self.figure_title, 
                             tools=[self.hover, SaveTool()])

        self.bar = self.figure.vbar(x="rank", 
                                    top="metric", 
                                    fill_color=self.color_map,
                                    name="cluster_bar_chart",
                                    source=self.source, 
                                    visible=True)
        return 

    def create_metric_df(self, selections):

        self.metric_df = self.text_viz_df.groupby(by="cluster")[[selections['cluster_bar_column_plotted']]].agg(func=["mean", "median", 
                                                                                                                      "min", "max", 
                                                                                                                      "sum", "std", 
                                                                                                                      "var", "sem", "size"])

        self.metric_df.columns = ['_'.join(col) for col in self.metric_df.columns.to_numpy()]

        #self.metric_df["cluster_size"] = self.text_viz_df["cluster"].value_counts()

        plot_key = f"{selections['cluster_bar_column_plotted']}_{selections['cluster_bar_metric_plotted']}"

        self.metric_df = self.metric_df.sort_values(by=plot_key, # TODO: Make _mean selectable
                                                    ascending=False).reset_index()


        self.metric_df = self.metric_df.reset_index().rename(columns={"index":"rank"})

        self.metric_df["metric"] = self.metric_df[plot_key].copy(deep=True)

        self.metric_df.columns = [col.split("_")[-1] for col in self.metric_df.columns.to_numpy()]

        return 