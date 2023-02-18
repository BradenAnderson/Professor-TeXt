import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, HoverTool, SaveTool,
                            Dropdown, Range1d, WheelZoomTool, ResetTool, PanTool)
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import  Plasma


class ResidualVsPredicted:
    def __init__(self, df,  color_settings, selections):

        self.df = df

        self.color_settings = None
        self.color_map = self._get_color_map(color_settings=color_settings)
        
        self.primary_key_column = selections['primary_key_column']
        self.axis_range_buffer_pct = selections["axis_range_buffer_pct"]

        self.dd_rvp_residual_column = None
        self.controls_layout = None

        self.scatter_size = 20
        self.rvp_scatter = None
        self.rvp_scatter_source = None

        self.tool_tips = self._get_tooltips(selections=selections)
        self.hover = HoverTool(tooltips=self.tool_tips, attachment="below")

        self.figure = None
        self.figure_title = None

    def initialize_figure_controls(self, selections):

        self.dd_rvp_residual_column = Dropdown(label=f"Residual Type = {selections['rvp_residuals_column']}", 
                                           name="select_rvp_resids", 
                                           button_type="success",
                                           menu=[("Externally Studentized", "externally_studentized_residuals"),  
                                                 ("Studentized", "student_resid"), 
                                                 ("PRESS Residuals", "press_residuals"), 
                                                 ("Internally Studentized", "internally_studentized_residuals"), 
                                                 ("Residuals", "residuals")])


        self.controls_layout = column(children=[self.dd_rvp_residual_column])    

        return 

    def _get_color_map(self, color_settings, field_name="cluster"):
        self.color_settings = color_settings
        return linear_cmap(field_name=field_name, 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])


    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)
        self.df['rvp_resid'] = self.df[selections['rvp_residuals_column']].to_numpy()
        self.rvp_scatter_source = ColumnDataSource(self.df)

        self.figure = figure(title=self.figure_title, 
                             tools=[self.hover, WheelZoomTool(), ResetTool(), PanTool(), SaveTool()], 
                             x_axis_label="Predicted",
                             y_axis_label="Residual",
                             width=800, 
                             height=500)

        self.rvp_scatter = self.figure.scatter(x="fitted_values", 
                                           y="rvp_resid", 
                                           name="residual_vs_predicted_plot", 
                                           alpha=0.5,
                                           size=self.scatter_size, 
                                           source=self.rvp_scatter_source, 
                                           fill_color=self.color_map)

        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.x_range = Range1d(start=x_min, 
                                      end=x_max, 
                                      name="rvp_xrange")

        self.figure.y_range = Range1d(start=y_min, 
                                      end=y_max, 
                                      name="rvp_yrange")

        self.initialize_figure_controls(selections=selections)

        return 

    def _get_x_y_axis_ranges(self):
        x_range = self.df['fitted_values'].max() - self.df['fitted_values'].min()
        y_range = self.df['rvp_resid'].max() - self.df['rvp_resid'].min()
        x_min = self.df['fitted_values'].min() - x_range*self.axis_range_buffer_pct
        x_max = self.df['fitted_values'].max() + x_range*self.axis_range_buffer_pct
        y_min = self.df['rvp_resid'].min() - y_range*self.axis_range_buffer_pct
        y_max = self.df['rvp_resid'].max() + y_range*self.axis_range_buffer_pct
        return x_min, x_max, y_min, y_max

    def update_figure(self, df, selections):
        
        # Update title
        self.figure.title.text = self.get_figure_title(selections=selections)

        # Update data source
        self.df = df
        self.df['rvp_resid'] = self.df[selections['rvp_residuals_column']].to_numpy()

        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.select({'name':'rvp_xrange'}).update(start=x_min, end=x_max)
        self.figure.select({'name':'rvp_yrange'}).update(start=y_min, end=y_max)

        self.rvp_scatter_source.data = ColumnDataSource.from_df(self.df)

        # Update marker colors
        self.color_map = self._get_color_map(color_settings=selections['current_cluster_cmap_info'])
        self.figure.select({'name':'residual_vs_predicted_plot'}).glyph.update(fill_color=self.color_map)

        return 

    def get_figure_title(self, selections):

        t = f"{selections['rvp_residuals_column']} vs predicted value"
        return t

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