import numpy as np
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import (Button, TabPanel, TextAreaInput, NumericInput, Dropdown, PlainText,
TextInput, Div, DataTable, TableColumn, ColumnDataSource, NumberFormatter, StringFormatter)
from bokeh.layouts import column, row, grid
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma

from .regression_scatter import MLRScatter
from .qq_plot import QQPlot
from .resid_vs_leverage import ResidualVsLeverage
from .resid_vs_predicted import ResidualVsPredicted
from .resid_histogram import ResidualHistogram

from ..backend.linear_regression import MLRByCluster
from ..backend.regression_lines_data_generator import MlrLinesGenerator

class RegressionTab:
    def __init__(self, primary_key_column):
        
        
        self.primary_key_column = primary_key_column
        self.selections = None
        self.text_viz_df = None
        
        self.mlr = None             # instance of MLRByCluster
        self.line_generator = None  # instance of MlrLinesGenerator

        self.mlr_scatter = None
        self.qqplot = None
        self.resid_vs_leverage = None
        self.resid_vs_predicted = None
        self.resid_histogram = None

        self.parameter_estimate_table = None  # DataTable
        self.parameter_estimate_df = None     # MLRByCluster.params_df
        self.parameter_estimate_source = None # ColumnDataSource

        self.regression_metrics_table = None  # DataTable
        self.regression_metrics_df = None     # MLRByCluster.params_df
        self.regression_metrics_source = None # ColumnDataSource

        self.regression_controls = None
        
        self.regression_model_controls = None
        self.button_reg_settings = None
        self.dd_reg_type = None
        self.dd_reg_x_column = None
        self.numeric_input_obs_removal = None 

        self.regression_prediction_controls = None
        self.button_reg_predict = Button(label="Submit Prediction", button_type="primary")
        self.numeric_input_reg_x_predict = None
        self.dd_reg_cluster_predict = None
        self.text_input_reg_cluster_predict = None

        return

    def get_prediction(self, x_values, category_values):

        X = self.mlr.preprocess_future_predictions(x_values=x_values, category_values=category_values)

        y_preds = self.mlr.model.predict(X).to_numpy().ravel()

        if y_preds.shape[0] == 1:
            y_preds = y_preds[0]

        return y_preds

    def _initialize_regression_controls(self):

        self._initialize_regression_model_controls()
        self._initialize_regression_prediction_controls()

        self.regression_controls = row(children=[self.regression_prediction_controls, self.regression_model_controls]) 

        return

    def _initialize_regression_prediction_controls(self):

        self.button_reg_predict = Button(label="Submit Prediction", 
                                         button_type="primary", 
                                         width=self.selections['regression_controls_width'], 
                                         height=self.selections['regression_controls_height'])

        self.div_predicted_value = Div(text="Predicted Value: ", 
                                       render_as_text=True, 
                                       name="display_reg_prediction", 
                                       width=self.selections['regression_controls_width'], 
                                       height=self.selections['regression_controls_height'])

        self.numeric_input_reg_x_predict = NumericInput(placeholder=f"Enter X value for Prediction", 
                                                        width=self.selections['regression_controls_width'], 
                                                        height=self.selections['regression_controls_height'])

        self.dd_reg_cluster_predict = Dropdown(label=f"Enter Cluster for Prediction", 
                                               name="select_reg_cluster_predict", 
                                               button_type="success",
                                               menu=[(f"Cluster {cluster_number}", f"{cluster_number}") for cluster_number in 
                                                      sorted(self.text_viz_df['cluster'].unique().tolist())], 
                                               width=self.selections['regression_controls_width'], 
                                               height=self.selections['regression_controls_height']) 

        self.text_input_reg_cluster_predict = TextAreaInput(title="Ad Text for Prediction", 
                                                            rows=20, 
                                                            sizing_mode="fixed",
                                                            width=self.selections['regression_controls_width'])

        self.regression_prediction_controls = column(children=[self.button_reg_predict, 
                                                               self.div_predicted_value,
                                                               self.numeric_input_reg_x_predict, 
                                                               self.dd_reg_cluster_predict, 
                                                               column(children=[self.text_input_reg_cluster_predict],
                                                                      sizing_mode="stretch_height")])
        return 
        
    def _initialize_regression_model_controls(self):

        self.button_reg_settings = Button(label="Submit Model Settings", 
                                          button_type="primary", 
                                         width=self.selections['regression_controls_width'], 
                                         height=self.selections['regression_controls_height'])

        self.dd_reg_type = Dropdown(label=f"Regression Type = {self.selections['mlr_model_type']}", 
                                           name="select_regression_type", 
                                           button_type="success",
                                           menu=[("Simple Linear Regression", "slr"),  
                                                 ("Multiple Linear Regression (Parallel)", "parallel_lines"), 
                                                 ("Multiple Linear Regression (Separate)", "separate_lines")],
                                                 width=self.selections['regression_controls_width'], 
        height=self.selections['regression_controls_height']) 

        
        self.dd_reg_x_column = Dropdown(label=f"Regression X Column = {self.selections['mlr_x_column']}", 
                                           name="select_regression_x_column", 
                                           button_type="success",
                                           menu=[self.selections['mlr_x_column']] +\
                                                    [(f"{col.title()}", col) for col in self.selections['numeric_metric_columns'] 
                                                        if col != self.selections['mlr_x_column']], 
                                           width=self.selections['regression_controls_width'], 
                                           height=self.selections['regression_controls_height'])

        self.dd_reg_y_column = Dropdown(label=f"Regression y Column = {self.selections['mlr_y_column']}", 
                                           name="select_regression_y_column", 
                                           button_type="success",
                                           menu=[self.selections['mlr_y_column']] +\
                                                    [(f"{col.title()}", col) for col in self.selections['numeric_metric_columns'] 
                                                        if col != self.selections['mlr_y_column']], 
                                           width=self.selections['regression_controls_width'], 
                                           height=self.selections['regression_controls_height'])


        self.numeric_input_obs_removal = TextInput(name="regression_outlier_observation_numbers", 
                                                       title="Observation Numbers To Remove (Comma Separated)",
                                                       value="", 
                                                       width=self.selections['regression_controls_width'], 
                                                       height=self.selections['regression_controls_height']*4)


        self.dd_target_transformations = Dropdown(label=f"Target Transformation = {self.selections['mlr_y_column_transformation']}", 
                                                  name="mlr_target_transformation", 
                                           button_type="success",
                                           menu=[("None", "None"), ("Log Transform", "log_transform")], 
                                           width=self.selections['regression_controls_width'], 
                                           height=self.selections['regression_controls_height'])

        self.regression_model_controls = column(children=[self.button_reg_settings, 
                                                          self.dd_reg_type, 
                                                          self.dd_reg_x_column, 
                                                          self.dd_reg_y_column, 
                                                          self.dd_target_transformations,
                                                          column(children=[self.numeric_input_obs_removal], 
                                                                 sizing_mode="stretch_height")]) 
                                                
        return 

    def setup_plots(self):

        self.run_mlr()

        self.initialize_plots()

        return 

    def initialize_plots(self):
        
        # MLR 
        print("init mlr scatter")
        self.mlr_scatter = MLRScatter(df=self.text_viz_df, 
                                      line_data=self.line_generator.data_sources, 
                                      color_settings=self.selections['current_cluster_cmap_info'], 
                                      selections=self.selections)
        self.mlr_scatter.initialize_figure(selections=self.selections)

        # Params table
        print("init params table")
        self._initialize_parameter_estimates_table()

        # Metrics table
        print("init metrics table")
        self._initialize_regression_metrics_table()

        # QQ
        print("init qq")
        self.qqplot = QQPlot(df=self.text_viz_df, 
                             color_settings=self.selections['current_cluster_cmap_info'], 
                             reference_line_type=self.selections['qq_reference_line_type'], 
                             selections=self.selections)
        self.qqplot.initialize_figure(selections=self.selections)

        # Resid vs Leverage
        print("init resid v leverage")
        self.resid_vs_leverage = ResidualVsLeverage(df=self.text_viz_df, 
                                                    color_settings=self.selections['current_cluster_cmap_info'], 
                                                    selections=self.selections)
        self.resid_vs_leverage.initialize_figure(selections=self.selections)

        self._initialize_regression_controls()

        # Resid vs Predicted 
        print("init resid v pred")
        self.resid_vs_predicted = ResidualVsPredicted(df=self.text_viz_df, 
                                                      color_settings=self.selections['current_cluster_cmap_info'], 
                                                      selections=self.selections)
        self.resid_vs_predicted.initialize_figure(selections=self.selections)

        # Resid Histogram
        print("init resid hist")
        self.resid_histogram = ResidualHistogram(df=self.text_viz_df)
        self.resid_histogram.initialize_figure(selections=self.selections)

        return 

    def _initialize_regression_metrics_table(self):

        self.regression_metrics_df = self.mlr.metrics_df
        self.regression_metrics_source = ColumnDataSource(self.regression_metrics_df)

        table_columns = [TableColumn(field="  ", 
                                     title="", 
                                     formatter=StringFormatter()), 
                        TableColumn(field=" ", 
                                     title="", 
                                     formatter=NumberFormatter(format=self.selections['regression_table_number_format'])), 
                        TableColumn(field="   ", 
                                     title="", 
                                     formatter=StringFormatter()), 
                        TableColumn(field="", 
                                     title="", 
                                     formatter=NumberFormatter(format=self.selections['regression_table_number_format']))]

        self.regression_metrics_table = DataTable(width=500, 
                                                  height=350, 
                                                  columns=table_columns,
                                                  name="regression_metrics_chart",
                                                  source=self.regression_metrics_source)
        
        return 

    def _initialize_parameter_estimates_table(self):

        self.parameter_estimate_df = self.mlr.params_df
        self.parameter_estimate_source = ColumnDataSource(self.parameter_estimate_df)

        numeric_columns = self.parameter_estimate_df.select_dtypes(include=[np.number]).columns.tolist()
        string_columns = [c for c in self.parameter_estimate_df.columns if c not in numeric_columns]
        table_columns = [TableColumn(field=col, 
                                     title=col.title(), 
                                     visible=True, 
                                     formatter=NumberFormatter(format=self.selections['regression_table_number_format'])) for col in numeric_columns]
        table_columns = [TableColumn(field=col, 
                                     title=col.title(), 
                                     visible=True, 
                                     formatter=StringFormatter()) for col in string_columns] + table_columns

        self.parameter_estimate_table = DataTable(width=500, 
                                                  height=350, 
                                                  columns=table_columns,
                                                  name="regression_parameters_chart",
                                                  source=self.parameter_estimate_source)
        
        return 

    def update_plots(self):
        
        if self.selections["mlr_observations_to_remove"] and isinstance(self.selections["mlr_observations_to_remove"], list):
            self.text_viz_df = self.text_viz_df.loc[~self.text_viz_df[self.primary_key_column].isin(self.selections["mlr_observations_to_remove"]),:].copy(deep=True).reset_index(drop=True)

        self.run_mlr()

        # Scatter and lines
        print("update mlr scatter")
        self.mlr_scatter.update_figure(df=self.text_viz_df, 
                                       line_data=self.line_generator.data_sources,
                                       selections=self.selections)

        # Param Table
        print("update param table")
        self.parameter_estimate_df = self.mlr.params_df                            
        self.parameter_estimate_source.data = ColumnDataSource.from_df(self.parameter_estimate_df)

        # Metric Table
        print("update metric table")
        self.regression_metrics_df = self.mlr.metrics_df                            
        self.regression_metrics_source.data = ColumnDataSource.from_df(self.regression_metrics_df)

        # QQ Plot
        print("update qq")
        self.qqplot.update_figure(df=self.text_viz_df, 
                                  selections=self.selections)

        # Resid vs Leverage
        print("update resid v leverage")
        self.resid_vs_leverage.update_figure(df=self.text_viz_df, 
                                             selections=self.selections)

        # Resid vs Leverage
        print("update resid v pred")
        self.resid_vs_predicted.update_figure(df=self.text_viz_df, 
                                              selections=self.selections)

        # Resid Histogram
        print("update resid hist")
        self.resid_histogram.update_figure(df=self.text_viz_df, 
                                          selections=self.selections)

        return

    def _set_tab_layout(self):

        # TODO: Fix spacing in layout --> check out spacers

        tables = column(children=[Div(text="Regression Parameter Estimates", render_as_text=True),
                                  self.parameter_estimate_table, 
                                  Div(text="Regression Metrics", render_as_text=True),
                                  self.regression_metrics_table], sizing_mode="stretch_height")
        row1 = row(children=[self.regression_controls, self.mlr_scatter.figure, tables])

        row2 = row(children=[column(self.qqplot.controls_layout, self.qqplot.figure), 
                             column(self.resid_histogram.controls_layout, self.resid_histogram.figure),
                             column(self.resid_vs_predicted.controls_layout, self.resid_vs_predicted.figure),
                             column(self.resid_vs_leverage.controls_layout, self.resid_vs_leverage.figure)])

        layout = column(row1, row2)

        self.layout = TabPanel(child=layout, title="Regression")

        return

    def get_layout(self):

        self._set_tab_layout()

        return self.layout 


    def run_mlr(self):

        self.mlr = MLRByCluster(df=self.text_viz_df, 
                                x_column=self.selections['mlr_x_column'], 
                                y_column=self.selections['mlr_y_column'], 
                                categorical_column=self.selections['mlr_categorical_column'], 
                                model_type=self.selections['mlr_model_type'])
        
        self.mlr.fit()

        self.line_generator = MlrLinesGenerator(model=self.mlr.model, 
                                                df=self.text_viz_df, 
                                                x_column=self.selections['mlr_x_column'], 
                                                categorical_column=self.selections['mlr_categorical_column'], 
                                                model_type=self.selections['mlr_model_type'])
        
        self.line_generator.create_line_data()

        self.text_viz_df = self.mlr.df.copy(deep=True)

        return 

    def set_selections(self, selections):
        self.selections = selections
        return

    def set_text_viz_df(self, text_viz_df):
        self.text_viz_df = text_viz_df
        return