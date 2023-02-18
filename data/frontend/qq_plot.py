import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, Range1d, WheelZoomTool, ResetTool, PanTool, SaveTool
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import  Plasma

from statsmodels.graphics.gofplots import qqplot

class QQPlot:
    def __init__(self, df,  color_settings, selections, reference_line_type="45"):

        self.df = df

        self.color_settings = None
        self.color_map = self._get_color_map(color_settings=color_settings)
        self.axis_range_buffer_pct = selections["axis_range_buffer_pct"]
        self.primary_key_column = selections['primary_key_column']

        self.reference_line_source = None 
        self.reference_line_type = reference_line_type
        self.reference_line = None

        self.dd_qq_reference_linetype = None
        self.dd_qq_residual_column = None
        self.controls_layout = None

        self.scatter_size = 20
        self.qq = None
        self.qq_source = None        

        self.tool_tips = self._get_tooltips(selections=selections)
        self.hover = HoverTool(tooltips=self.tool_tips, attachment="below")

        self.figure = None
        self.figure_title = None

    def initialize_figure_controls(self, selections):

        self.dd_qq_residual_column = Dropdown(label=f"Residual Type = {selections['qq_residuals_column']}", 
                                           name="select_qq_resids", 
                                           button_type="success",
                                           menu=[("Externally Studentized", "externally_studentized_residuals"),  
                                                 ("Studentized", "student_resid"), 
                                                 ("PRESS Residuals", "press_residuals"), 
                                                 ("Internally Studentized", "internally_studentized_residuals"), 
                                                 ("Residuals", "residuals")])

        self.dd_qq_reference_linetype = Dropdown(label=f"Reference Line Type = {selections['qq_reference_line_type']}", 
                                           name="select_qq_reference_linetype", 
                                           button_type="success",
                                           menu=[("45-degrees", "45"),  
                                                 ("Standardized Line", "s"), 
                                                 ("Regression Line", "r"), 
                                                 ("Through Quartiles", "q"),
                                                 ("No Reference Line", "None")])

        self.controls_layout = column(children=[self.dd_qq_residual_column, self.dd_qq_reference_linetype])    

        return 

    def _setup_data_sources(self, selections):
        
        self.df = self.df.sort_values(by=selections['qq_residuals_column'])

        qq_sm = qqplot(data=self.df[selections['qq_residuals_column']], fit=True, line=self.reference_line_type)

        qq_scatter_data = self._get_qq_data(qq_sm=qq_sm)
        self.df['qq_x'] = qq_scatter_data['x']
        self.df['qq_y'] = qq_scatter_data['y']
        self.qq_source = ColumnDataSource(self.df)

        qq_reference_line_data = self._get_reference_line_data(qq_sm=qq_sm)
        self.reference_line_source = ColumnDataSource(pd.DataFrame(qq_reference_line_data))
        
        return 

    def _get_qq_data(self, qq_sm):
        
        qq_scatter_data = qq_sm.axes[0].lines[0].get_xydata()
        qq_scatter_data = {'x':qq_scatter_data[:,0], "y":qq_scatter_data[:,1]}
        return qq_scatter_data

    def _get_reference_line_data(self, qq_sm):

        if self.reference_line_type is not None:
            qq_ref_line_data = qq_sm.axes[0].lines[1].get_xydata()
            qq_ref_line_data = {'x':qq_ref_line_data[:,0].ravel(), 
                                'y':qq_ref_line_data[:,1].ravel()}
        else:
            qq_ref_line_data = {'x':[], 
                                'y':[]}
        return qq_ref_line_data

    def _get_color_map(self, color_settings, field_name="cluster"):
        self.color_settings = color_settings
        return linear_cmap(field_name=field_name, 
                                    palette=color_settings['palette'], 
                                    low=color_settings['low'], 
                                    high=color_settings['high'])


    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)
        self._setup_data_sources(selections=selections)

        self.figure = figure(title=self.figure_title, 
                             tools=[self.hover, WheelZoomTool(), ResetTool(), PanTool(), SaveTool()], 
                             width=500, 
                             height=500)

        self.figure.yaxis.axis_label = "Sample Quantiles"
        self.figure.xaxis.axis_label = "Theoretical Quantiles"
        
        self.qq = self.figure.scatter(x="qq_x", 
                                      y="qq_y", 
                                      name="qq_plot", 
                                      alpha=0.5,
                                      size=self.scatter_size, 
                                      source=self.qq_source, 
                                      fill_color=self.color_map)

        self.reference_line = self.figure.line(x="x", 
                                               y="y", 
                                               name=f"qq_reference_line",
                                               line_color="#000000", 
                                               source=self.reference_line_source)


        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.x_range = Range1d(start=x_min, 
                                      end=x_max, 
                                      name="qq_xrange")

        self.figure.y_range = Range1d(start=y_min, 
                                      end=y_max, 
                                      name="qq_yrange")

        self.initialize_figure_controls(selections=selections)

        return 

    def _get_x_y_axis_ranges(self):
        qq_df = self.qq_source.to_df()
        x_range = qq_df['qq_x'].max() - qq_df['qq_x'].min()
        y_range = qq_df['qq_y'].max() - qq_df['qq_y'].min()
        x_min = qq_df['qq_x'].min() - x_range*self.axis_range_buffer_pct
        x_max = qq_df['qq_x'].max() + x_range*self.axis_range_buffer_pct
        y_min = qq_df['qq_y'].min() - y_range*self.axis_range_buffer_pct
        y_max = qq_df['qq_y'].max() + y_range*self.axis_range_buffer_pct
        return x_min, x_max, y_min, y_max

    def update_figure(self, df, selections):
        self.reference_line_type = selections['qq_reference_line_type']
        self.figure.title.text = self.get_figure_title(selections=selections)
        self.df = df
        
        self.color_map = self._get_color_map(color_settings=selections['current_cluster_cmap_info'])
        self.figure.select({'name':'qq_plot'}).glyph.update(fill_color=self.color_map)

        ## Update x and y columns in self.scatter?? --> copy data to correct column name
        self._update_data_sources(selections=selections)
        x_min, x_max, y_min, y_max = self._get_x_y_axis_ranges()
        self.figure.select({'name':'qq_xrange'}).update(start=x_min, end=x_max)
        self.figure.select({'name':'qq_yrange'}).update(start=y_min, end=y_max)
        return 

    def _update_data_sources(self, selections):

        self.df = self.df.sort_values(by=selections['qq_residuals_column'])

        qq_sm = qqplot(data=self.df[selections['qq_residuals_column']], fit=True, line=self.reference_line_type)
        qq_scatter_data = self._get_qq_data(qq_sm=qq_sm)
        self.df['qq_x'] = qq_scatter_data['x']
        self.df['qq_y'] = qq_scatter_data['y']
        qq_reference_line_data = self._get_reference_line_data(qq_sm=qq_sm)
        self.qq_source.data = ColumnDataSource.from_df(self.df)
        self.reference_line_source.data = ColumnDataSource.from_df(pd.DataFrame(qq_reference_line_data))

        return

    def get_figure_title(self, selections):

        t = f"QQ Plot of {selections['qq_residuals_column']}"
        if self.reference_line_type is not None:
            t += f", Reference Line Type: {self.reference_line_type}"

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