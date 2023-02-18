import numpy as np
import pandas as pd 
import statsmodels.api as sm 

class MlrLinesGenerator:
    def __init__(self, model, df, x_column="Spend", categorical_column="cluster", 
                 model_type="separate_lines"):
        
        self.model = model # Fitted OLS model from Statsmodels
        self.model_type = model_type # parallel_lines, separate_lines, slr
        
        self.df = df
        self.x_column = x_column
        self.categorical_column = categorical_column

        self.model_input_columns = self.model.params.index.tolist()
        self.model_input_columns.remove("const")
        self.X = None 
        
        # For creating prediction dataframe, used to draw the 
        # parallel or separate lines model
        self.metric_min = self.df[self.x_column].min()
        self.metric_max = self.df[self.x_column].max()
        self.num_samples = 1_000
        self.num_clusters = None
        self.cluster_numbers = None
        self._set_num_categories()
        
        self.X_plot = None
        self.data_sources = None
    
    def create_line_data(self):
        
        self._get_data()
        
        self._setup_data_sources()
        
        return 
    
    def _setup_data_sources(self):
        
        if self.model_type == "slr":
            self.data_sources = {0:self.X_plot.copy(deep=True)}
        elif self.model_type == "parallel_lines" or self.model_type == "separate_lines":
            self._add_cluster_column_to_X_plot()
            self.data_sources = {num:self.X_plot.loc[self.X_plot["cluster"]==num,:].copy(deep=True) for num in self.cluster_numbers}
            
    def _add_cluster_column_to_X_plot(self):

        self.X_plot['cluster'] = [cluster_number for cluster_number in self.cluster_numbers
                                      for _ in range(self.num_samples)]
        return 
    
    def _get_data(self):
        
        # Set up dataframe with samples evenly spaced across the domain of x_column
        # We predict on this and then use the predictions to draw the lines for each cluster
        self._setup_model_inputs()
        
        self.X_plot = self.X.copy(deep=True)
        
        self.X_plot.rename(columns={self.x_column:"X_plot"}, inplace=True)

        self.X_plot = self.X_plot.loc[:, ["X_plot", "y_plot"]]
        
        return
    
    def _setup_slr_inputs(self):
        
        # Create self.num_samples of the model input, evenly spaced between the 
        # smallest and largest observed values
        samples = self._get_exog_domain_samples()
        
        self.X = pd.DataFrame({"X_plot":np.squeeze(samples)})
        
        return
    
    def _setup_parallel_lines_inputs(self):
        
        # Create self.num_samples of self.x_column, evenly spaced between the 
        # smallest and largest observed values
        samples = self._get_exog_domain_samples()
        
        # Matrix of one-hot encoded cluster columns, shape is (n_samples*n_clusters-1, n_clusters-1)
        # -1 is because we drop one of the one-hot encoded cluster columns (reference category)
        # This gives n_samples observations from each cluster category
        # These are the intercept adjustment terms
        cluster_ohe_matrix = self._get_manual_ohe_matrix()
        
        self.X = pd.DataFrame(np.concatenate([samples, cluster_ohe_matrix], axis=1), 
                              columns=self.model_input_columns)
        
        return 
    
    def _setup_separate_lines_inputs(self):
        
        # Create self.num_samples of self.x_column, evenly spaced between the 
        # smallest and largest observed values
        samples = self._get_exog_domain_samples()

        intercept_adjustment_matrix = self._get_manual_ohe_matrix()

        slope_adjustment_matrix = np.multiply(samples, intercept_adjustment_matrix)

        self.X = pd.DataFrame(np.concatenate([samples, 
                                              intercept_adjustment_matrix, 
                                              slope_adjustment_matrix], 
                                             axis=1), 
                              columns=self.model_input_columns)
        return 
    
    def _setup_model_inputs(self):
        
        if self.model_type == "slr":
            self._setup_slr_inputs()
        elif self.model_type == "parallel_lines":
            self._setup_parallel_lines_inputs()
        elif self.model_type == "separate_lines":
            self._setup_separate_lines_inputs()
        
        self.X = sm.add_constant(self.X)
        
        self.X["y_plot"] = self.model.predict(self.X)
        
        self.X.drop(columns="const", inplace=True)
        
        return 
    
    def _get_exog_domain_samples(self):
        
        samples = np.linspace(start=self.metric_min, 
                              stop=self.metric_max, 
                              num=self.num_samples).tolist()
        
        all_samples = np.array(samples*self.num_clusters).reshape(-1, 1)
        
        return all_samples
    
    def _get_manual_ohe_column(self, input_shape, hot_index):
        zeros = np.zeros(shape=input_shape)
        zeros[:, hot_index] = 1
        return zeros
    
    def _get_manual_ohe_matrix(self):
        
        input_shape = (self.num_samples, self.num_clusters-1)
        
        hot_arrays = [self._get_manual_ohe_column(input_shape=input_shape, 
                                            hot_index=num) 
              for num in range(0, self.num_clusters-1)]
        
        all_zeros = np.zeros(input_shape)
        
        manual_ohe_matrix = np.concatenate([all_zeros]+hot_arrays, axis=0)

        return manual_ohe_matrix
    
    def _set_num_categories(self):
        if self.model_type == "slr":
            self.num_clusters = 1
            self.cluster_numbers = [0]
        else:
            self.num_clusters = self.df[self.categorical_column].nunique()
            self.cluster_numbers = [num for num in range(self.num_clusters)]