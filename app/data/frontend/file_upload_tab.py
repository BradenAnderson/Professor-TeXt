import numpy as np
import pandas as pd

from bokeh.models import Button, TabPanel, FileInput, TextInput, Div, Dropdown, Switch
from bokeh.layouts import column, row

class FileUploadTab:
    def __init__(self):
        
        self.selections = None 
        
        self.div_run_with_sample_data = Div(text="<b>Run with sample data (Off for file upload):</b>") 
        self.switch_run_with_sample_data = None 

        self.submit_button_file_upload =  None
        
        self.div_file_upload = Div(text="Select a .csv file to upload: ", 
                                   visible=False)
        self.file_upload = None

        self.div_text_column = Div(text="Select column containing text:", 
                                   visible=False)
        self.dd_text_column = None

        numeric_txt = ("Input names of numeric columns (comma separated)<br>" 
                       "to use for predictions or aggregation statistics:")
        self.div_numeric_metric_columns = Div(text=numeric_txt,
                                              visible=False)
        self.text_input_numeric_metric_columns = None

        self.div_hover_columns = Div(text="Input names of columns to show in hover tools (comma separated):", 
                                     visible=False)
        self.text_input_hover_columns = None

        self.controls = None
        
        return

    def initialize_controls(self):

        self.switch_run_with_sample_data = Switch(active=True)

        self.submit_button_file_upload = Button(label="Submit Application Settings", 
                                                button_type="success", 
                                                visible=True)


        self.file_upload = FileInput(accept=".csv", visible=False)

        self.dd_text_column = Dropdown(label="Select Text Containing Column", button_type="primary",
                                       menu=[], 
                                       visible=False)
        
        self.text_input_numeric_metric_columns = TextInput(title="", value="", visible=False)
        self.text_input_hover_columns = TextInput(title="", value="", visible=False)

        self.controls = column(row(self.div_run_with_sample_data, self.switch_run_with_sample_data), 
                               self.submit_button_file_upload, 
                               column(self.div_file_upload, self.file_upload), 
                               column(self.div_text_column, self.dd_text_column), 
                               column(self.div_numeric_metric_columns, self.text_input_numeric_metric_columns), 
                               column(self.div_hover_columns, self.text_input_hover_columns))

    def _set_tab_layout(self):

        self.layout = TabPanel(child=self.controls, title="App Settings")

        return

    def get_layout(self):

        self._set_tab_layout()

        return self.layout 

    def set_selections(self, selections):
        self.selections = selections
        return