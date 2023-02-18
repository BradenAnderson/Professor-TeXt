import numpy as np
import pandas as pd 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

class MLRByCluster:
    def __init__(self, df, y_column="Cost per 1k impressions", x_column="Spend", 
                 categorical_column="cluster", model_type="separate_lines"):
        
        self.model_type = model_type # parallel_lines, separate_lines
        self.df = df
        self.x_column = x_column
        self.y_column = y_column
        self.categorical_column = categorical_column
        self.polynomial_features = None # Sklearns Polynomial features if using separate_lines
        self.ohe = None

        self.X = self.df[[self.x_column]].copy(deep=True)
        self.y = self.df[self.y_column]
        
        self.model = None 
        self.params_df = None 
        self.outlier_influence = None
        self.metrics_df = None 

    
    def fit(self):
        
        self._preprocess()
        
        self.model = sm.OLS(endog=self.y, exog=self.X).fit()

        self._postprocess()
        
        return 
    
    def preprocess_future_predictions(self, x_values, category_values):
        """Used to preprocess data for predictions asked for in the app"""

        self.prediction_data = pd.DataFrame({self.x_column:x_values if isinstance(x_values, list) else [x_values]})
        if self.model_type == "slr":
            self.prediction_data = sm.add_constant(self.prediction_data)
            return self.prediction_data

        cluster_pred_df = pd.DataFrame({self.categorical_column:category_values 
                                                if isinstance(category_values, list) else [category_values]})
        
        if self.model_type == "parallel_lines" or self.model_type == "separate_lines":
            encoded_df = self.ohe.transform(cluster_pred_df[self.categorical_column].to_numpy().reshape(-1, 1))
            encoded_df = encoded_df.rename(columns={f:f"cluster_{f.split('_')[-1]}" for f in self.ohe.get_feature_names_out()})
            encoded_df = encoded_df.drop(columns="cluster_0")

        self.prediction_data = pd.concat(objs=[self.prediction_data, encoded_df], axis="columns")

        if self.model_type == "separate_lines":
            prediction_poly = self.polynomial_features.transform(self.prediction_data)
            prediction_poly = prediction_poly.loc[:, [c for c in prediction_poly.columns 
                                                    if self.x_column in c and "cluster" in c]]
            self.prediction_data = pd.concat(objs=[self.prediction_data, prediction_poly], axis="columns")
        const_df = pd.DataFrame({'const':[1 for _ in range(self.prediction_data.shape[0])]})
        self.prediction_data = pd.concat(objs=[const_df, self.prediction_data], axis="columns")

        return self.prediction_data

    def _preprocess(self):
        
        if self.model_type == "slr":
            self.X = self.df[[self.x_column]]
        
        if self.model_type == "parallel_lines" or self.model_type == "separate_lines":
            self._add_intercept_adjustment_columns_from_clusters()
        
        if self.model_type == "separate_lines":
            self._add_slope_adjustment_terms()
        
        # Intercept term
        self.X = sm.add_constant(self.X)
        
        return 
    
    def _add_intercept_adjustment_columns_from_clusters(self):
        
        """One hot encoder categorical_column (clusters from NLP tab
        by default) and adds it to self.X"""
        
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe.set_output(transform="pandas")
        cluster_df = self.ohe.fit_transform(self.df[self.categorical_column].to_numpy().reshape(-1, 1))
        cluster_df = cluster_df.rename(columns={f:f"cluster_{f.split('_')[-1]}" for f in self.ohe.get_feature_names_out()})
        cluster_df = cluster_df.drop(columns="cluster_0")
        self.X = pd.concat(objs=[self.X, cluster_df], axis="columns")
        return 
    
    def _add_slope_adjustment_terms(self):
        
        self.polynomial_features = PolynomialFeatures(include_bias=False, interaction_only=True)
        self.polynomial_features.set_output(transform="pandas")
        X_poly = self.polynomial_features.fit_transform(self.X)
        X_poly = X_poly.loc[:, [c for c in X_poly.columns if self.x_column in c and "cluster" in c]]
        self.X = pd.concat(objs=[self.X, X_poly], axis="columns")
        return
    
    def _postprocess(self):
        
        self._set_params_df()

        self._set_metrics_df()
        
        self._calculate_outlier_influence_statistics()

        self.df['x'] = self.df[self.x_column].to_numpy()
        self.df['y'] = self.df[self.y_column].to_numpy()
        
        if self.df.isna().sum().sum() > 0:
            msg = ("\n\nWARNING: NaN values found after running linear regression. "
                   "This may be a statsmodels problem related to near perfect fits (residuals = 0)... Are you "
                   "running separate or parallel lines models and have one or more very small groups? These "
                   "NaN values are being manually set to zero, proceed with caution!\n\n")
            print(msg)
            self.df = self.df.fillna(0)

        return 
    
    def _calculate_outlier_influence_statistics(self):

        self.outlier_influence = OLSInfluence(results=self.model)

        outlier_summary = self.outlier_influence.summary_frame()
        outlier_summary['externally_studentized_residuals'] = self.outlier_influence.get_resid_studentized_external()
        outlier_summary['press_residuals'] = self.outlier_influence.resid_press.to_numpy()
        outlier_summary['internally_studentized_residuals'] = self.outlier_influence.resid_studentized_internal
        outlier_summary.rename(columns={"hat_diag":"leverage"}, inplace=True)
        self.df = pd.concat(objs=[self.df, outlier_summary], axis="columns")

        return

    def _set_metrics_df(self):
        metrics = {"  ":["No. Observations:", "DF Model:", "DF Residuals:", 
                        "F-Statistic:", "F-Test P-Value:", "Explained SOS:", "RSS:", 
                        "TSS (Cenetered):", "TSS (Uncentered):"], 
                " ":[self.model.nobs, self.model.df_model, self.model.df_resid,
                        self.model.fvalue, self.model.f_pvalue, self.model.ess, self.model.ssr, 
                        self.model.centered_tss, self.model.uncentered_tss], 
                "   ":["R-Squared:", "Adjusted R-Squared:", "AIC:", "BIC:", 
                        "Log-Likelihood:", "MSE Model:", "MSE Residual:", 
                        "MSE Total:", "ESS PRESS: "],
                "":[self.model.rsquared, self.model.rsquared_adj, self.model.aic, self.model.bic, 
                    self.model.llf, self.model.mse_model, self.model.mse_resid, 
                    self.model.mse_total, self.model.get_influence().ess_press]}
        
        self.metrics_df = pd.DataFrame(metrics)

        return

    def _set_params_df(self):
        
        params_df = pd.DataFrame({"parameter":self.model.params.index.tolist(),
                                  "estimate":self.model.params.to_numpy(), 
                                  "p_value":self.model.pvalues.to_numpy()}, 
                                  index=self.model.params.index.tolist())
        ci_df = self.model.conf_int(alpha=0.05).rename(columns={0:"lower_ci", 1:"upper_ci"})
        self.params_df = pd.concat(objs=[params_df, ci_df], axis="columns").loc[:, ["parameter","estimate", "lower_ci", "upper_ci", "p_value"]]
        
        self.df['fitted_values'] = self.model.fittedvalues
        self.df['residuals'] = self.model.resid
        
        return 