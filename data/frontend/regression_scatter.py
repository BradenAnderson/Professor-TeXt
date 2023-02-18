import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d, WheelZoomTool, ResetTool, PanTool, SaveTool
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import  Plasma

from ..backend.linear_regression import MLRByCluster
from ..backend.regression_lines_data_generator import MlrLinesGenerator

class MLRScatter:
    def __init__(self, df, line_data, color_settings, selections):

        self.df = df
        self.line_data = line_data
        self.primary_key_column = selections['primary_key_column']

        self.color_settings = None
        self.color_map = self._get_color_map(color_settings=color_settings)
        self.axis_range_buffer_pct = selections["axis_range_buffer_pct"]
        self.scatter_size = 20
        self.scatter_source = None
        self.line_sources = None

        self.tool_tips = self._get_tooltips(selections=selections)
        self.hover = HoverTool(tooltips=self.tool_tips, attachment="below")

        self.scatter = None
        self.lines = None

        self.figure = None
        self.figure_title = None



    def initialize_data_source(self, selections):
        self.df['mlr_x_column'] = self.df[selections['mlr_x_column']].to_numpy()
        self.df['mlr_y_column'] = self.df[selections['mlr_y_column']].to_numpy()
        self.scatter_source = ColumnDataSource(self.df)
        self.line_sources = {cluster:ColumnDataSource(data) for cluster, data 
                                in self.line_data.items()} 
        return

    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)
        self.initialize_data_source(selections=selections)

        self.figure = figure(title=self.figure_title, 
                             tools=[self.hover, WheelZoomTool(), ResetTool(), PanTool(), SaveTool()], 
                             width=selections["regression_scatter_chart_width"], 
                             height=selections["regression_scatter_chart_height"])

        self.scatter = self.figure.scatter(x='mlr_x_column', 
                                           y='mlr_y_column', 
                                           name="regression_scatter_chart", 
                                           alpha=0.5,
                                           size=self.scatter_size, 
                                           source=self.scatter_source, 
                                           fill_color=self.color_map)
        self.figure.hover.renderers = [self.scatter]
        self.figure.yaxis.axis_label = selections['mlr_y_column']
        self.figure.xaxis.axis_label = selections['mlr_x_column']
        
        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.x_range = Range1d(start=x_min, 
                                      end=x_max, 
                                      name="mlr_scatter_xrange")

        self.figure.y_range = Range1d(start=y_min, 
                                      end=y_max, 
                                      name="mlr_scatter_yrange")

        self.lines = {cluster:self.figure.line(x="X_plot", 
                      y="y_plot", 
                      name=f"mlr_line_cluster_{cluster}",
                      line_color=self.color_settings['palette'][cluster],
                      source=source) for cluster, source in self.line_sources.items()}

        # self.initialize_figure_controls()

        return 

    def update_figure(self, df, line_data, selections):

        self.figure.title.text = self.get_figure_title(selections=selections)
        self.figure.yaxis.axis_label = selections['mlr_y_column']
        self.figure.xaxis.axis_label = selections['mlr_x_column']

        self.df = df
        self.line_data = line_data
        self.df['mlr_x_column'] = self.df[selections['mlr_x_column']].to_numpy()
        self.df['mlr_y_column'] = self.df[selections['mlr_y_column']].to_numpy()

        self.color_map = self._get_color_map(color_settings=selections['current_cluster_cmap_info'])
        self.figure.select({'name':'regression_scatter_chart'}).glyph.update(fill_color=self.color_map)

        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.select({'name':'mlr_scatter_xrange'}).update(start=x_min, end=x_max)
        self.figure.select({'name':'mlr_scatter_yrange'}).update(start=y_min, end=y_max)
        self.update_data_sources()

        return 

    def update_data_sources(self):
        
        self.scatter_source.data = ColumnDataSource.from_df(self.df)
        ## Update scatter to use current palette

        current_clusters = list(self.line_data.keys())
        previous_seen_cluster_numbers = list(self.line_sources.keys())

        # For each cluster in the most recent set of lines from MlrLinesGenerator
        for cluster, dataframe in self.line_data.items():
            
            # If there is no key for this cluster number in self.line_sources
            if cluster not in previous_seen_cluster_numbers:

                # Create new entry in dict mapping cluster # --> Data Source
                self.line_sources[cluster] = ColumnDataSource(dataframe)

                # Add a glyph to the figure and store in the dict mapping cluster # --> MLR line
                self.lines[cluster] = self.figure.line(x="X_plot", 
                                                       y="y_plot", 
                                                       name=f"mlr_line_cluster_{cluster}",
                                                       line_color=self.color_settings['palette'][cluster],
                                                       source=self.line_sources[cluster])
            
            else: # Else update the data in the data source for this MLR Line
                self.figure.select({"name":f"mlr_line_cluster_{cluster}"}).glyph.update(line_color=self.color_settings['palette'][cluster])
                self.line_sources[cluster].data = ColumnDataSource.from_df(dataframe)
        
        # Turn off lines no longer being used
        for cluster, line in self.lines.items():
            if cluster not in current_clusters:
                line.visible = False
            else:
                line.visible = True
        

        return 

    def get_figure_title(self, selections):

        m_type = selections['mlr_model_type']
        x = selections['mlr_x_column']
        y = selections['mlr_y_column']
        categorical = selections['mlr_categorical_column']

        model_type_map = {'separate_lines': f"Multiple Linear Regression (Separate Lines) of {y} on {x}", 
                        'parallel_lines': f"Multiple Linear Regression (Parallel Lines) of {y} on {x}", 
                        "slr": f"Simple Linear Regression of {y} on {x}"}

        reg = f"{model_type_map[m_type]}"

        if m_type != "slr":
            reg += f" for each level of {categorical}"

        return reg

    def _get_x_y_axis_ranges(self):
        x_range = self.df['mlr_x_column'].max() - self.df['mlr_x_column'].min()
        y_range = self.df['mlr_y_column'].max() - self.df['mlr_y_column'].min()
        x_min = self.df['mlr_x_column'].min() - x_range*self.axis_range_buffer_pct
        x_max = self.df['mlr_x_column'].max() + x_range*self.axis_range_buffer_pct
        y_min = self.df['mlr_y_column'].min() - y_range*self.axis_range_buffer_pct
        y_max = self.df['mlr_y_column'].max() + y_range*self.axis_range_buffer_pct
        return x_min, x_max, y_min, y_max

    def _get_color_map(self, color_settings, field_name="cluster"):
        self.color_settings = color_settings
        return linear_cmap(field_name=field_name, 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])

    def _get_tooltips(self, selections):
        
        TOOLTIPS = (
            '<div>'
                f'<div style="width:{selections["hover_window_width"]};">'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Text: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{Txt}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Cluster: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{cluster}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Observation Number: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{{self.primary_key_column}}}</span><br>'
                '</div>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">X: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{x}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">y: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{y}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Fitted Value: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{fitted_values}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Cooks D: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{cooks_d}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">DFFITS: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{dffits}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Externally Studentized Residual: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{externally_studentized_residuals}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Studentized Residual: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{student_resid}}</span><br>'
                '</div>'
                '<div>'
                f'<span style="font-size: {selections["hover_text_size"]}; font-weight: bold;">Leverage: </span>'
                f'<span style="font-size: {selections["hover_text_size"]};">@{{leverage}}</span><br>'
                '</div>'
                '<br>'
            '</div>'
        )

        return TOOLTIPS