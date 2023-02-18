import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, LinearAxis, Range1d, SaveTool
from bokeh.layouts import column, row

import scipy.stats as stats


class ResidualHistogram:
    def __init__(self, df):

        self.df = df
        self.num_bins = None
        self.hist_df = None

        self.dd_rhist_residual_column = None
        self.controls_layout = None

        self.resid_histogram = None
        self.resid_histogram_source = None

        self.normal_curve = None
        self.normal_curve_source = None

        self.tool_tips = [("Bin Count", "@{bin_count}"), 
                          ("Description", "@{Description}"), 
                          ("Bin Probability", "@{bin_probability}")]
        self.hover = HoverTool(tooltips=self.tool_tips)

        self.figure = None
        self.figure_title = None

    def initialize_figure_controls(self, selections):

        self.dd_rhist_residual_column = Dropdown(label=f"Residual Type = {selections['rvp_residuals_column']}", 
                                           name="select_rvp_resids", 
                                           button_type="success",
                                           menu=[("Externally Studentized", "externally_studentized_residuals"),  
                                                 ("Studentized", "student_resid"), 
                                                 ("PRESS Residuals", "press_residuals"), 
                                                 ("Internally Studentized", "internally_studentized_residuals"), 
                                                 ("Residuals", "residuals")])


        self.controls_layout = column(children=[self.dd_rhist_residual_column])    

        return 

    def _set_normal_curve_data(self, selections):

        resid_array = self.df[selections['rhist_residuals_column']].to_numpy()
        resid_std = np.std(resid_array)

        normal = stats.norm(loc=0, 
                            scale=resid_std)

        x_samples = np.linspace(start=np.min(resid_array), stop=np.max(resid_array), num=200)    
        y_samples = np.array([normal.pdf(x) for x in x_samples])

        normal_curve_data = {'x':x_samples, 
                             'y':y_samples}

        self.normal_curve_df = pd.DataFrame(normal_curve_data)
        #self.normal_curve_df['Normal Curve (Resid) Std'] = resid_std

        return 

    def initialize_figure(self, selections):

        self.figure_title = self.get_figure_title(selections=selections)

        self._initialize_data_source(selections=selections)

        self.figure = figure(title=self.figure_title, 
                             x_axis_label="Residual",
                             y_axis_label="Bin Count", 
                             tools=[self.hover, SaveTool()],
                             width=800, 
                             height=500)

        self.resid_histogram = self.figure.quad(bottom=0, 
                                                top='bin_count',
                                                left='left_edge', 
                                                right='right_edge',
                                                fill_color="#0B1354", 
                                                source=self.resid_histogram_source, 
                                                name="residual_histogram")
        self.figure.hover.renderers = [self.resid_histogram]
        self.figure.yaxis.axis_label = "Bin Count"
        self.figure.xaxis.axis_label = "Residual"

        self.figure.extra_y_ranges = {'y2': Range1d(start=0,
                                                    end=self.normal_curve_df['y'].max()+0.05*self.normal_curve_df['y'].max(), 
                                                    name="normal_curve_yrange")}

        self.figure.add_layout(obj=LinearAxis(y_range_name="y2", 
                            axis_label="Normal Distribution Probability"),
             place="right")

        
        self.normal_curve = self.figure.line(x="x", 
                                             y='y', 
                                             y_range_name='y2', 
                                             source=self.normal_curve_source,  
                                             color="#F765A3")

        self.initialize_figure_controls(selections=selections)

        return 

    def _initialize_data_source(self, selections):

        self._setup_data_source(selections=selections)

        self.resid_histogram_source = ColumnDataSource(self.hist_df)
        self.normal_curve_source = ColumnDataSource(self.normal_curve_df)

        return
    
    def _update_data_source(self, selections):

        self._setup_data_source(selections=selections)

        self.resid_histogram_source.data = ColumnDataSource.from_df(self.hist_df)
        self.normal_curve_source.data = ColumnDataSource.from_df(self.normal_curve_df)

        self.figure.select({'name':'normal_curve_yrange'}).update(end=self.normal_curve_df['y'].max()+0.05*self.normal_curve_df['y'].max())
        return

    def _setup_data_source(self, selections):

        self.num_bins = selections["rhist_residuals_num_bins"]
        self.df['rhist_resid'] = self.df[selections['rhist_residuals_column']].to_numpy()
        resid_array = self.df['rhist_resid'].to_numpy()
        h1 = np.histogram(a=resid_array[~np.isnan(resid_array)], 
                          range=(resid_array[~np.isnan(resid_array)].min(), 
                                 resid_array[~np.isnan(resid_array)].max()), 
                          bins=self.num_bins, 
                          density=False)
        
        h2 = np.histogram(a=resid_array, 
                          range=(resid_array[~np.isnan(resid_array)].min(), 
                                 resid_array[~np.isnan(resid_array)].max()), 
                          bins=self.num_bins, 
                          density=True)

        hist_data = {'bin_count':h1[0], 
                     'bin_probability':h2[0],
                     'left_edge':h1[1][:-1], 
                     'right_edge':h1[1][1:]}
        
        self.hist_df = pd.DataFrame(hist_data)

        self.hist_df['Description'] = [f"Bin Min: {min_val:.3f}, Bin Max: {max_val:.3f}" 
                                            for min_val, max_val in zip(self.hist_df['left_edge'].to_numpy(),
                                                                        self.hist_df['right_edge'].to_numpy())]

        self._set_normal_curve_data(selections=selections)

        return 

    def update_figure(self, df, selections):
        
        # Update title
        self.figure.title.text = self.get_figure_title(selections=selections)

        # Update data source
        self.df = df
        #self.df['rhist_resid'] = self.df[selections['rhist_residuals_column']].to_numpy()
        self._update_data_source(selections=selections)

        return 

    def get_figure_title(self, selections):

        t = f"Histogram of {selections['rhist_residuals_column']}"
        return t