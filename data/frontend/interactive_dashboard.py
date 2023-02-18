import numpy as np
import pandas as pd

from bokeh.events import ButtonClick, MenuItemClick
from bokeh.plotting import curdoc

from base64 import b64decode
import io

from .dashboard import Dashboard 

class BokehDashboard:
    def __init__(self, selections, paths, df=None):

        self.bokeh_doc = curdoc()
        self.bokeh_doc.title = "Professor TeXt"
        
        self.initial_selections = selections
        self.tab_list = None

        self.dashboard = Dashboard(selections=self.initial_selections.copy(), 
                                   paths=paths, 
                                   df=df)

        return

    def setup(self):
        
        self.dashboard.setup_file_upload_tab()

        self.setup_app_settings_callbacks()

        self.tab_list = [self.dashboard.file_upload_panel]

        self.bokeh_doc.add_root(self.dashboard.tabs)

        return

    def setup_application(self):
        
        # Build and set up the dashboard elements
        self.dashboard.setup_dashboard()

        # Add interactivity to the controls
        self.setup_application_callbacks()

        return

    def setup_app_settings_callbacks(self):

        self.dashboard.file_upload_tab.switch_run_with_sample_data.on_change("active", self.switch_run_with_sample_data_callback)
        self.dashboard.file_upload_tab.submit_button_file_upload.on_event(ButtonClick, self.submit_button_file_upload_callback) 
        self.dashboard.file_upload_tab.file_upload.on_change("value", self.file_upload_callback)
        
        self.dashboard.file_upload_tab.dd_text_column.on_event(MenuItemClick, self.dd_text_column_callback)
        self.dashboard.file_upload_tab.text_input_numeric_metric_columns.on_change("value", self.text_input_numeric_metric_columns_callback)
        self.dashboard.file_upload_tab.text_input_hover_columns.on_change("value", self.text_input_hover_columns_callback)
    
    def switch_run_with_sample_data_callback(self, attr, old, new):

        self.dashboard.update_selections(run_with_sample_data=new)

        if new:
            self.dashboard.file_upload_tab.div_file_upload.update(visible=False) 
            self.dashboard.file_upload_tab.file_upload.update(visible=False) 
            self.dashboard.file_upload_tab.div_text_column.update(visible=False) 
            self.dashboard.file_upload_tab.dd_text_column.update(label="Select Text Containing Column", visible=False) 
            self.dashboard.file_upload_tab.div_numeric_metric_columns.update(visible=False) 
            self.dashboard.file_upload_tab.text_input_numeric_metric_columns.update(visible=False) 
            self.dashboard.file_upload_tab.div_hover_columns.update(visible=False) 
            self.dashboard.file_upload_tab.text_input_hover_columns.update(visible=False) 
        else:
            self.dashboard.file_upload_tab.submit_button_file_upload.update(visible=True)
            self.dashboard.file_upload_tab.div_file_upload.update(visible=True) 
            self.dashboard.file_upload_tab.file_upload.update(visible=True) 
            self.dashboard.file_upload_tab.div_text_column.update(visible=True) 
            self.dashboard.file_upload_tab.dd_text_column.update(visible=True) 
            self.dashboard.file_upload_tab.div_numeric_metric_columns.update(visible=True) 
            self.dashboard.file_upload_tab.text_input_numeric_metric_columns.update(visible=True) 
            self.dashboard.file_upload_tab.div_hover_columns.update(visible=True) 
            self.dashboard.file_upload_tab.text_input_hover_columns.update(visible=True) 

        return

    def text_input_hover_columns_callback(self, attr, old, new):

        always_include = ["cluster", self.dashboard.selections['primary_key_column']]
        cols = [col.strip() for col in new.split(",")]
        cols = always_include + cols

        self.dashboard.update_selections(embedding_scatter_hover_columns=cols)
        return

    def text_input_numeric_metric_columns_callback(self, attr, old, new):
        cols = [col.strip() for col in new.split(",")]
        self.dashboard.update_selections(numeric_metric_columns=cols)
        return

    def dd_text_column_callback(self, event):
        self.dashboard.file_upload_tab.dd_text_column.update(label=f"text_column={event.item}")
        self.dashboard.update_selections(text_column=event.item)
        return
    
    def submit_button_file_upload_callback(self, event):

        print("in submit button callback...")
        # If we are running with sample data, reset all selections to the initial settings.
        if self.dashboard.selections['run_with_sample_data']:
            self.dashboard.update_selections(**self.initial_selections)

        self.tab_list = [self.tab_list[0]]
        self.dashboard.tabs.update(tabs=self.tab_list)
        # del self.dashboard.nlp_tab 
        # self.dashboard.nlp_tab  = None
        # del self.dashboard.regression_tab
        # self.dashboard.regression_tab = None 
        # del self.dashboard.text_generation_tab
        # self.dashboard.text_generation_tab = None 

        # Ensure all column selections are valid for the new dataset (make sure that no selections
        # are using a column that does not exist in the new dataset)
        first_metric = self.dashboard.selections['numeric_metric_columns'][0]
        second_metric = self.dashboard.selections['numeric_metric_columns'][1]
        
        embed_hovers_always_include = ["cluster", self.dashboard.selections['primary_key_column']]
        embed_hovers = np.unique(embed_hovers_always_include + self.dashboard.selections['numeric_metric_columns']).tolist()
        self.dashboard.update_selections(cluster_bar_column_plotted=first_metric, 
                                         mlr_x_column=first_metric, 
                                         mlr_y_column=second_metric, 
                                         mlr_y_column_transformation=None, 
                                         text_gen_success_metric=first_metric, 
                                         embedding_scatter_hover_columns=embed_hovers)
        
        self.setup_application()
        
        self.tab_list.extend([self.dashboard.nlp_panel, self.dashboard.regression_panel, self.dashboard.text_generation_panel])
        self.dashboard.tabs.update(tabs=self.tab_list)
        print("Finished updating tabs!")
        
        return

    def file_upload_callback(self, attr, old, new):
        
        # Read selected file into pandas dataframe
        file_decoded = b64decode(new)
        file_bytes = io.BytesIO(file_decoded)
        df = pd.read_csv(file_bytes)
        self.dashboard.set_dataframe(df=df)

        # Find numeric and non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number, int, float]).columns.tolist()
        non_numeric_columns = [col for col in df.columns if col not in numeric_columns]

        # Add non-numeric columns to text_column dropdown menu
        self.dashboard.file_upload_tab.dd_text_column.update(menu=non_numeric_columns)

        # Populate metric and hover col text inputs with numeric columns
        non_numeric_col_string = ",".join(numeric_columns)
        self.dashboard.file_upload_tab.text_input_numeric_metric_columns.update(value=non_numeric_col_string)
        self.dashboard.file_upload_tab.text_input_hover_columns.update(value=non_numeric_col_string)
        self.dashboard.update_selections(numeric_metric_columns=numeric_columns, 
                                         embedding_scatter_hover_columns=numeric_columns)

        return

    def setup_application_callbacks(self):

        # Tab change
        self.dashboard.tabs.on_change('active',  self.tab_change_callback)

        # Embed
        self.dashboard.nlp_tab.embedding_scatter.dropdown_embedding.on_event(MenuItemClick, self.dropdown_embedding_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_hugging_model_checkpoint.on_event(MenuItemClick, self.dropdown_hugging_model_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_hugging_embedding_type.on_event(MenuItemClick, self.dropdown_hugging_embedding_type_callback)
        self.dashboard.nlp_tab.embedding_scatter.text_hugging_model_checkpoint.on_change('value', self.text_hugging_model_checkpoint_callback)

        # Embed - TFIDF Params 
        self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_strip_accents.on_event(MenuItemClick, self.dd_tfidf_strip_accents_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_norm.on_event(MenuItemClick, self.dd_tfidf_norm_callback)
        self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_min_df.on_change('active', self.radio_tfidf_min_df_callback)
        self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_max_df.on_change('active', self.radio_tfidf_max_df_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.on_change('value', self.slider_tfidf_max_df_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.on_change('value', self.slider_tfidf_min_df_callback)
        self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.on_change('value', self.spinner_tfidf_max_df_callback)
        self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.on_change('value', self.spinner_tfidf_min_df_callback)

        self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_lowercase.on_change('active', self.switch_tfidf_lowercase_callback)
        self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_binary.on_change('active', self.switch_tfidf_binary_callback)
        self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_use_idf.on_change('active', self.switch_tfidf_use_idf_callback)
        self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_smooth_idf.on_change('active', self.switch_tfidf_smooth_idf_callback)
        self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_sublinear_tf.on_change('active', self.switch_tfidf_sublinear_tf_callback)

        # Dim Reduce
        self.dashboard.nlp_tab.embedding_scatter.dropdown_dimreduce.on_event(MenuItemClick, self.dropdown_dimreduce_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_scale_before_dim_reduce.on_event(MenuItemClick, self.dd_scale_before_dim_reduce_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_step1_dimreduce_ndims.on_change('value', self.slider_step1_dimreduce_ndims_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_umap_metric.on_event(MenuItemClick, self.dropdown_umap_metric_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_tsne_perplexity.on_change('value', self.slider_tsne_perplexity_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_umap_n_neighbors.on_change('value', self.umap_neighbors_slider_callback)

        # Cluster
        self.dashboard.nlp_tab.embedding_scatter.dropdown_cluster.on_event(MenuItemClick, self.dropdown_cluster_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_cluster_space.on_event(MenuItemClick, self.dd_cluster_space_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_scale_before_cluster.on_event(MenuItemClick, self.dd_scale_before_cluster_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_n_clusters.on_change('value', self.cluster_slider_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_agg_affinity.on_event(MenuItemClick, self.dd_agglomerative_affinity_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_agg_linkage.on_event(MenuItemClick, self.dd_agglomerative_linkage_callback)
        self.dashboard.nlp_tab.embedding_scatter.slider_kmeans_ninit.on_change('value', self.kmeans_ninit_callback)
        self.dashboard.nlp_tab.embedding_scatter.dd_kmeans_algo.on_event(MenuItemClick, self.dd_kmeans_algo_callback)

        # Submit text viz settings
        self.dashboard.nlp_tab.submit_button.on_event(ButtonClick, self.submit_button_callback)

        # Metric bar plot
        self.dashboard.nlp_tab.cluster_barplot.cluster_bar_column_radio.on_change("active", self.bar_cluster_column_button_callback)
        self.dashboard.nlp_tab.cluster_barplot.cluster_bar_metric_radio.on_change("active", self.bar_cluster_metric_button_callback)

        # Regression Plot (Predictions)
        self.dashboard.regression_tab.button_reg_predict.on_event(ButtonClick, self.reg_prediction_button_callback)
        self.dashboard.regression_tab.numeric_input_reg_x_predict.on_change('value', self.reg_pred_numeric_x_input_callback)
        self.dashboard.regression_tab.dd_reg_cluster_predict.on_event(MenuItemClick, self.reg_dd_cluster_predict_callback)
        self.dashboard.regression_tab.text_input_reg_cluster_predict.on_change('value', self.reg_pred_ad_text_callback)

        # Regression Plot (Model settings)
        self.dashboard.regression_tab.button_reg_settings.on_event(ButtonClick, self.reg_model_button_callback)
        self.dashboard.regression_tab.dd_reg_type.on_event(MenuItemClick, self.reg_dd_model_type_callback)
        self.dashboard.regression_tab.dd_reg_x_column.on_event(MenuItemClick, self.reg_dd_model_x_column_callback)
        self.dashboard.regression_tab.dd_reg_y_column.on_event(MenuItemClick, self.reg_dd_model_y_column_callback)
        self.dashboard.regression_tab.dd_target_transformations.on_event(MenuItemClick, self.dd_target_transformations_callback)
        self.dashboard.regression_tab.numeric_input_obs_removal.on_change('value', self.reg_model_obs_to_remove_callback)

        # QQ
        self.dashboard.regression_tab.qqplot.dd_qq_residual_column.on_event(MenuItemClick, self.dd_qq_residual_column_callback)
        self.dashboard.regression_tab.qqplot.dd_qq_reference_linetype.on_event(MenuItemClick, self.dd_qq_reference_linetype_callback)

        # RVL
        self.dashboard.regression_tab.resid_vs_leverage.dd_rvl_residual_column.on_event(MenuItemClick, self.dd_rvl_residual_column_callback)

        # RVP
        self.dashboard.regression_tab.resid_vs_predicted.dd_rvp_residual_column.on_event(MenuItemClick, self.dd_rvp_residual_column_callback)

        # Resid Hist
        self.dashboard.regression_tab.resid_histogram.dd_rhist_residual_column.on_event(MenuItemClick, self.dd_rhist_residual_column_callback)

        ### Text Generation
        self.dashboard.text_generation_tab.button_generate_tweet.on_event(ButtonClick, self.submit_button_generate_tweet_callback)
        self.dashboard.text_generation_tab.switch_autogenerate_prompt.on_change('active', self.switch_autogenerate_prompt_callback)
        self.dashboard.text_generation_tab.switch_use_recent_tweets.on_change('active', self.switch_use_recent_tweets_callback) 
        self.dashboard.text_generation_tab.switch_use_success_examples.on_change('active', self.switch_use_success_examples_callback)
        
        self.dashboard.text_generation_tab.text_input_twitter_account_name.on_change('value', self.text_input_twitter_account_name_callback)
        self.dashboard.text_generation_tab.text_input_tweet_topic.on_change('value', self.text_input_tweet_topic_callback)
        self.dashboard.text_generation_tab.text_input_prompt.on_change('value', self.text_input_prompt_callback)

        self.dashboard.text_generation_tab.dd_input_tweet_tone.on_event(MenuItemClick, self.dd_input_tweet_tone_callback)
        self.dashboard.text_generation_tab.dd_success_metric.on_event(MenuItemClick, self.dd_success_metric_callback)
        self.dashboard.text_generation_tab.dd_openai_model.on_event(MenuItemClick, self.dd_openai_model_callback)

        self.dashboard.text_generation_tab.spinner_num_recent_examples.on_change('value', self.spinner_num_recent_examples_callback)
        self.dashboard.text_generation_tab.spinner_num_success_examples.on_change('value', self.spinner_num_success_examples_callback)
        self.dashboard.text_generation_tab.spinner_openai_max_tokens.on_change('value', self.spinner_openai_max_tokens_callback)

        self.dashboard.text_generation_tab.slider_openai_temperature.on_change('value', self.slider_openai_temperature_callback)
        self.dashboard.text_generation_tab.slider_openai_top_p.on_change('value', self.slider_openai_top_p_callback)
        self.dashboard.text_generation_tab.ms_tweets_to_exclude.on_change('active', self.ms_tweets_to_exclude_callback)

        ### Cosine Similarity Matrix
        self.dashboard.nlp_tab.cosine_similarity_matrix.switch_filter_sim_matrix.on_change('active', self.switch_filter_sim_matrix_callback)

        self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_filter_type.on_event(MenuItemClick, self.dd_sim_matrix_filter_type_callback)
        self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.on_event(MenuItemClick, self.dd_sim_matrix_focus_type_callback)

        self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_filter_size.on_change('value', self.spinner_sim_matrix_filter_size_callback)
        self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_focus_id.on_change('value', self.spinner_sim_matrix_focus_id_callback)

        self.dashboard.nlp_tab.cosine_similarity_matrix.text_input_sim_matrix_filter_exact_ids.on_change('value', self.text_input_sim_matrix_filter_exact_ids_callback)

        self.dashboard.nlp_tab.cosine_similarity_matrix.button_submit_filter.on_event(ButtonClick, self.button_submit_filter_callback)
        return 

    def button_submit_filter_callback(self, event):
        self.dashboard.nlp_tab.cosine_similarity_matrix.update_figure(selections=self.dashboard.selections)
        return

    def spinner_sim_matrix_filter_size_callback(self, attr, old, new):
        self.dashboard.update_selections(sim_matrix_filter_size=new)
        return 

    def spinner_sim_matrix_focus_id_callback(self, attr, old, new):
        self.dashboard.update_selections(sim_matrix_filter_focus_id=str(new))
        return 
        
    def text_input_sim_matrix_filter_exact_ids_callback(self, attr, old, new):
        self.dashboard.update_selections(sim_matrix_filter_exact_ids=[str(val) for val in new.split(",")])
        return

    def switch_filter_sim_matrix_callback(self, attr, old, new):
        self.dashboard.update_selections(sim_matrix_filter=new)

        if new: 

            self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_filter_type.update(visible=True)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_type.update(visible=True)
            
            if self.dashboard.selections['sim_matrix_filter_type'] in ['random', 'focused']:
                self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_filter_size.update(visible=True)
                self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_size.update(visible=True)
            
            if self.dashboard.selections['sim_matrix_filter_type'] == "focused":
                self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.update(visible=True)
                self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_type.update(visible=True)
                self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_focus_id.update(visible=True)
                self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_id.update(visible=True)

            if self.dashboard.selections['sim_matrix_filter_type'] == "exact":
                self.dashboard.nlp_tab.cosine_similarity_matrix.text_input_sim_matrix_filter_exact_ids.update(visible=True) 
                self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_exact_ids.update(visible=True)

            self.dashboard.nlp_tab.cosine_similarity_matrix.button_submit_filter.update(visible=True)
        else:
            self.dashboard.nlp_tab.cosine_similarity_matrix.button_submit_filter.update(visible=False)

            self.dashboard.nlp_tab.cosine_similarity_matrix.text_input_sim_matrix_filter_exact_ids.update(visible=False) 
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_exact_ids.update(visible=False)

            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_focus_id.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_id.update(visible=False)

            self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_type.update(visible=False)

            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_filter_size.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_size.update(visible=False)

            self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_filter_type.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_type.update(visible=False)

        return

    def dd_sim_matrix_filter_type_callback(self, event):
        self.dashboard.update_selections(sim_matrix_filter_type=event.item)

        new_label = f"filter_type={event.item}"
        self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_filter_type.update(label=new_label)

        if event.item in ["random", "focused"]:
            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_filter_size.update(visible=True)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_size.update(visible=True)
        else:
            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_filter_size.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_size.update(visible=False)

        if event.item == "focused":
            self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.update(visible=True)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_type.update(visible=True)
            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_focus_id.update(visible=True)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_id.update(visible=True)
        else:
            self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_type.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.spinner_sim_matrix_focus_id.update(visible=False)
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_focus_id.update(visible=False)

        if event.item == "exact":
            self.dashboard.nlp_tab.cosine_similarity_matrix.text_input_sim_matrix_filter_exact_ids.update(visible=True) 
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_exact_ids.update(visible=True)
        else:
            self.dashboard.nlp_tab.cosine_similarity_matrix.text_input_sim_matrix_filter_exact_ids.update(visible=False) 
            self.dashboard.nlp_tab.cosine_similarity_matrix.div_sim_matrix_filter_exact_ids.update(visible=False)

        return

    def dd_sim_matrix_focus_type_callback(self, event):
        self.dashboard.update_selections(sim_matrix_filter_focus_type=event.item)
        new_label = f"focus_type={event.item}"
        self.dashboard.nlp_tab.cosine_similarity_matrix.dd_sim_matrix_focus_type.update(label=new_label)
        return

    def ms_tweets_to_exclude_callback(self, attr, old, new):
        labels = self.dashboard.text_generation_tab.ms_tweets_to_exclude.labels 
        new_labels = [label for index, label in enumerate(labels) if index in new]
        self.dashboard.update_selections(text_gen_tweets_to_exclude=new_labels if new_labels != [] else None)
        return

    def slider_openai_temperature_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_temperature=new)
        return

    def slider_openai_top_p_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_top_p=new)
        return

    def spinner_openai_max_tokens_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_max_tokens=new)
        return 

    def spinner_num_recent_examples_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_num_recent_tweet_examples=new)
        return

    def spinner_num_success_examples_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_num_successful_tweet_examples=new)
        return 

    def dd_input_tweet_tone_callback(self, event):
        self.dashboard.update_selections(text_gen_tweet_tone=event.item)
        self.dashboard.text_generation_tab.dd_input_tweet_tone.update(label=f"tweet_tone={event.item}")
        return

    def dd_success_metric_callback(self, event):
        self.dashboard.update_selections(text_gen_success_metric=event.item)
        self.dashboard.text_generation_tab.dd_success_metric.update(label=f"metric={event.item}")
        return

    def dd_openai_model_callback(self, event):
        self.dashboard.update_selections(text_gen_model_name=event.item)
        self.dashboard.text_generation_tab.dd_openai_model.update(label=f"model={event.item}")
        return

    def text_input_tweet_topic_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_tweet_topic=new)
        return

    def text_input_twitter_account_name_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_twitter_username=new, text_gen_run_tweet_retriever=True)
        return

    def text_input_prompt_callback(self, attr, old, new):
        self.dashboard.update_selections(text_gen_prompt=new)
        return

    def submit_button_generate_tweet_callback(self, event):

        self.dashboard.text_generation_tab.update()

        self.dashboard.text_generation_tab.prompt_used_paragraph.update(text=self.dashboard.text_generation_tab.tweet_generator.prompt.replace("\n", "<br>"))

        self.dashboard.text_generation_tab.tweet_generated_paragraph.update(text=self.dashboard.text_generation_tab.generated_tweet)

        return

    def switch_autogenerate_prompt_callback(self, attr, old, new):

        if new: 
            self.dashboard.text_generation_tab.div_twitter_account_name.update(visible=True)
            self.dashboard.text_generation_tab.text_input_twitter_account_name.update(visible=True)

            self.dashboard.text_generation_tab.div_tweet_topic.update(visible=True)
            self.dashboard.text_generation_tab.text_input_tweet_topic.update(visible=True)

            self.dashboard.text_generation_tab.div_tweet_tone.update(visible=True)
            self.dashboard.text_generation_tab.dd_input_tweet_tone.update(visible=True)

            self.dashboard.text_generation_tab.div_use_recent_tweets.update(visible=True)
            self.dashboard.text_generation_tab.switch_use_recent_tweets.update(visible=True)

            self.dashboard.text_generation_tab.div_num_recent_tweets.update(visible=True)
            self.dashboard.text_generation_tab.spinner_num_recent_examples.update(visible=True)

            self.dashboard.text_generation_tab.div_tweets_to_exclude.update(visible=True)
            self.dashboard.text_generation_tab.ms_tweets_to_exclude.update(visible=True)

            self.dashboard.text_generation_tab.div_use_success_examples.update(visible=True)
            self.dashboard.text_generation_tab.switch_use_success_examples.update(visible=True)

            self.dashboard.text_generation_tab.div_num_success_examples.update(visible=True)
            self.dashboard.text_generation_tab.spinner_num_success_examples.update(visible=True)

            self.dashboard.text_generation_tab.div_success_metric.update(visible=True)
            self.dashboard.text_generation_tab.dd_success_metric.update(visible=True)

            self.dashboard.text_generation_tab.div_input_prompt.update(visible=False)
            self.dashboard.text_generation_tab.text_input_prompt.update(visible=False)

            self.dashboard.text_generation_tab.text_input_prompt.update(value="write your prompt here!")

        else:
            self.dashboard.text_generation_tab.div_twitter_account_name.update(visible=False)
            self.dashboard.text_generation_tab.text_input_twitter_account_name.update(visible=False)

            self.dashboard.text_generation_tab.div_tweet_topic.update(visible=False)
            self.dashboard.text_generation_tab.text_input_tweet_topic.update(visible=False)

            self.dashboard.text_generation_tab.div_tweet_tone.update(visible=False)
            self.dashboard.text_generation_tab.dd_input_tweet_tone.update(visible=False)

            self.dashboard.text_generation_tab.div_use_recent_tweets.update(visible=False)
            self.dashboard.text_generation_tab.switch_use_recent_tweets.update(visible=False)

            self.dashboard.text_generation_tab.div_num_recent_tweets.update(visible=False)
            self.dashboard.text_generation_tab.spinner_num_recent_examples.update(visible=False)

            self.dashboard.text_generation_tab.div_tweets_to_exclude.update(visible=False)
            self.dashboard.text_generation_tab.ms_tweets_to_exclude.update(visible=False)

            self.dashboard.text_generation_tab.div_use_success_examples.update(visible=False)
            self.dashboard.text_generation_tab.switch_use_success_examples.update(visible=False)

            self.dashboard.text_generation_tab.div_num_success_examples.update(visible=False)
            self.dashboard.text_generation_tab.spinner_num_success_examples.update(visible=False)

            self.dashboard.text_generation_tab.div_success_metric.update(visible=False)
            self.dashboard.text_generation_tab.dd_success_metric.update(visible=False)

            self.dashboard.text_generation_tab.div_input_prompt.update(visible=True)
            self.dashboard.text_generation_tab.text_input_prompt.update(visible=True)

            self.dashboard.update_selections(text_gen_prompt=None)

        return 

    def switch_use_recent_tweets_callback(self, attr, old, new):

        if new:

            self.dashboard.text_generation_tab.div_num_recent_tweets.update(visible=True)
            self.dashboard.text_generation_tab.spinner_num_recent_examples.update(visible=True)

            self.dashboard.text_generation_tab.div_tweets_to_exclude.update(visible=True)
            self.dashboard.text_generation_tab.ms_tweets_to_exclude.update(visible=True)
            
            self.dashboard.update_selections(text_gen_run_tweet_retriever=True, text_gen_use_recent_tweets=True)

        else:

            self.dashboard.text_generation_tab.div_num_recent_tweets.update(visible=False)
            self.dashboard.text_generation_tab.spinner_num_recent_examples.update(visible=False)

            self.dashboard.text_generation_tab.div_tweets_to_exclude.update(visible=False)
            self.dashboard.text_generation_tab.ms_tweets_to_exclude.update(visible=False)

            self.dashboard.update_selections(text_gen_run_tweet_retriever=False, text_gen_use_recent_tweets=False)
        
        return

    def switch_use_success_examples_callback(self, attr, old, new):

        if new:

            self.dashboard.text_generation_tab.div_num_success_examples.update(visible=True)
            self.dashboard.text_generation_tab.spinner_num_success_examples.update(visible=True)

            self.dashboard.text_generation_tab.div_success_metric.update(visible=True)
            self.dashboard.text_generation_tab.dd_success_metric.update(visible=True)

            self.dashboard.update_selections(text_gen_use_success_tweets=True)

        else:

            self.dashboard.text_generation_tab.div_num_success_examples.update(visible=False)
            self.dashboard.text_generation_tab.spinner_num_success_examples.update(visible=False)

            self.dashboard.text_generation_tab.div_success_metric.update(visible=False)
            self.dashboard.text_generation_tab.dd_success_metric.update(visible=False)

            self.dashboard.update_selections(text_gen_use_success_tweets=False)

        return 

    ##### Tab changing #####
    def tab_change_callback(self, attr, old, new):
        if new == 0:
            self.dashboard.update_selections(current_tab="app_settings")
        elif new == 1:
            self.dashboard.update_selections(current_tab="nlp")
        elif new == 2: 
            print("Switching to regression tab")
            self.dashboard.update_selections(current_tab="regression", 
                                             current_cluster_cmap_info=self.dashboard.nlp_tab.current_colors)
            self.dashboard.update_regression_tab()
        elif new == 3:
            self.dashboard.update_selections(current_tab="text_generation")
            
        return 

    ##### Regression Model #####
    def reg_model_button_callback(self, event):
        self.dashboard.regression_tab.div_predicted_value.update(text=f"Predicted Value: ")

        if self.dashboard.selections['mlr_y_column_transformation'] == "log_transform":

            # Create the log transformed column name
            target_column = self.dashboard.selections['mlr_y_column']
            log_column = f"Log {target_column}"

            # Perform log transform
            self.dashboard.nlp_tab.text_viz_df[log_column] = np.log(self.dashboard.nlp_tab.text_viz_df[target_column].to_numpy())

            # Update current regression target to log transformed version
            self.dashboard.selections.update(mlr_y_column=log_column)
        else:
            clean_column = self.dashboard.selections['mlr_y_column'].replace("Log ", "", 1)
            self.dashboard.selections.update(mlr_y_column=clean_column)

        self.dashboard.update_regression_tab()
        return

    def reg_dd_model_type_callback(self, event):
        self.dashboard.regression_tab.dd_reg_type.update(label=f"Regression Type = {event.item}")
        self.dashboard.update_selections(mlr_model_type=event.item)
        return 

    def reg_dd_model_x_column_callback(self, event):
        self.dashboard.regression_tab.dd_reg_x_column.update(label=f"Regression X Column = {event.item}")
        self.dashboard.update_selections(mlr_x_column=event.item)
        return 

    def reg_dd_model_y_column_callback(self, event):
        self.dashboard.regression_tab.dd_reg_y_column.update(label=f"Regression y Column = {event.item}")
        self.dashboard.update_selections(mlr_y_column=event.item)
        return 

    def dd_target_transformations_callback(self, event):
        self.dashboard.regression_tab.dd_target_transformations.update(label=f"Target Transformation = {event.item}")
        self.dashboard.update_selections(mlr_y_column_transformation=event.item)
        return


    def reg_model_obs_to_remove_callback(self, attr, old, new):
        if new:
            obs = [int(val) for val in new.split(",")]
        else:
            obs = None
        self.dashboard.update_selections(mlr_observations_to_remove = obs)
        return 

    ##### Regression Prediction #####
    def reg_prediction_button_callback(self, event):

        if self.dashboard.selections['mlr_ad_text_predict'] is not None:

            cluster_assignment = self.dashboard.nlp_tab.text_generator.predict([self.dashboard.selections['mlr_ad_text_predict']])

            self.dashboard.update_selections(mlr_cluster_predict=cluster_assignment)

            self.dashboard.regression_tab.dd_reg_cluster_predict.update(label=f"Cluster = {cluster_assignment}")
        else:
            cluster_assignment = self.dashboard.selections['mlr_cluster_predict']

        y_pred = self.dashboard.regression_tab.get_prediction(x_values=self.dashboard.selections['mlr_X_predict'], 
                                                              category_values=cluster_assignment)

    
        self.dashboard.regression_tab.div_predicted_value.update(text=f"Predicted {self.dashboard.selections['mlr_y_column']}: {y_pred}")                                                 
        return

    def reg_pred_numeric_x_input_callback(self, attr, old, new):
        self.dashboard.update_selections(mlr_X_predict = int(new))
        return 

    def reg_dd_cluster_predict_callback(self, event):
        self.dashboard.regression_tab.dd_reg_cluster_predict.update(label=f"Cluster = {event.item}")
        self.dashboard.update_selections(mlr_cluster_predict = int(event.item))
        return 

    def reg_pred_ad_text_callback(self, attr, old, new):
        self.dashboard.update_selections(mlr_ad_text_predict = new)
        return 

    ##### QQ Plot #####
    def dd_qq_residual_column_callback(self, event):
        self.dashboard.regression_tab.qqplot.dd_qq_residual_column.update(label=f"Residual Type = {event.item}")
        self.dashboard.update_selections(qq_residuals_column = event.item)
        self.dashboard.regression_tab.qqplot.update_figure(df=self.dashboard.regression_tab.text_viz_df, 
                                                           selections=self.dashboard.selections)
        return

    def dd_qq_reference_linetype_callback(self, event):
        self.dashboard.regression_tab.qqplot.dd_qq_reference_linetype.update(label=f"Reference Line Type = {event.item}")
        self.dashboard.update_selections(qq_reference_line_type = event.item if event.item != "None" else None)
        self.dashboard.regression_tab.qqplot.update_figure(df=self.dashboard.regression_tab.text_viz_df, 
                                                           selections=self.dashboard.selections)
        return

    ##### Residual vs Leverage #####
    def dd_rvl_residual_column_callback(self, event):
        self.dashboard.regression_tab.resid_vs_leverage.dd_rvl_residual_column.update(label=f"Residual Type = {event.item}")
        self.dashboard.update_selections(rvl_residuals_column = event.item) 
        self.dashboard.regression_tab.resid_vs_leverage.update_figure(df=self.dashboard.regression_tab.text_viz_df, 
                                                                      selections=self.dashboard.selections)
        return 


    ##### Residual vs Predicted #####
    def dd_rvp_residual_column_callback(self, event):
        self.dashboard.regression_tab.resid_vs_predicted.dd_rvp_residual_column.update(label=f"Residual Type = {event.item}")
        self.dashboard.update_selections(rvp_residuals_column = event.item) 
        self.dashboard.regression_tab.resid_vs_predicted.update_figure(df=self.dashboard.regression_tab.text_viz_df, 
                                                                      selections=self.dashboard.selections)
        return 

    ##### Residual Histogram #####
    def dd_rhist_residual_column_callback(self, event):
        self.dashboard.regression_tab.resid_histogram.dd_rhist_residual_column.update(label=f"Residual Type = {event.item}")
        self.dashboard.update_selections(rhist_residuals_column = event.item) 
        self.dashboard.regression_tab.resid_histogram.update_figure(df=self.dashboard.regression_tab.text_viz_df, 
                                                                      selections=self.dashboard.selections)
        return 

    ##### Embedding #####
    def dropdown_embedding_callback(self, event):
        
        if event.item == "hugging_face":
            self.dashboard.nlp_tab.embedding_scatter.dd_hugging_model_checkpoint.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.dd_hugging_embedding_type.update(visible=True) # how hidden states are combined to make a doc vector
            self.dashboard.nlp_tab.embedding_scatter.text_hugging_model_checkpoint.update(visible=True)
        else:
            self.dashboard.nlp_tab.embedding_scatter.dd_hugging_model_checkpoint.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.dd_hugging_embedding_type.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.text_hugging_model_checkpoint.update(visible=False)


        if event.item == "tfidf":
            self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_strip_accents.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_norm.update(visible=True)

            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_max_df.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_max_df.update(visible=True)
            if self.dashboard.selections['tfidf_specify_max_df_as'] == "proportion":
                self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.update(visible=True)
                self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.update(visible=False)
            else:
                self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.update(visible=False)
                self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.update(visible=True)

            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_min_df.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_min_df.update(visible=True)
            if self.dashboard.selections['tfidf_specify_min_df_as'] == "proportion":
                self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.update(visible=True)
                self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.update(visible=False)
            else:
                self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.update(visible=False)
                self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.update(visible=True)

            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_lowercase.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_lowercase.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_binary.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_binary.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_use_idf.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_use_idf.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_smooth_idf.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_smooth_idf.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_sublinear_tf.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_sublinear_tf.update(visible=True)

        else:
            self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_strip_accents.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_norm.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_max_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_max_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_min_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_min_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.update(visible=False)

            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_lowercase.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_lowercase.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_binary.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_binary.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_use_idf.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_use_idf.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_smooth_idf.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_smooth_idf.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.switch_tfidf_sublinear_tf.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.div_tfidf_sublinear_tf.update(visible=False)

        self.dashboard.nlp_tab.embedding_scatter.dropdown_embedding.update(label=f"Embedding = {event.item}")
        self.dashboard.update_selections(embedding_type=event.item)
        return 

    def dropdown_hugging_model_callback(self, event):
        self.dashboard.update_selections(hugging_face_model_checkpoint=event.item)
        new_label = f"HuggingFace Model = {event.item}"
        self.dashboard.nlp_tab.embedding_scatter.dd_hugging_model_checkpoint.update(label=new_label)
        return 
        
    def text_hugging_model_checkpoint_callback(self, attr, old, new):
        self.dashboard.update_selections(hugging_face_model_checkpoint = new)
        return 

    def dropdown_hugging_embedding_type_callback(self, event):

        self.dashboard.update_selections(hugging_face_embedding_type = event.item)
        new_label = f"Hidden State Embedding={event.item}"
        self.dashboard.nlp_tab.embedding_scatter.dd_hugging_embedding_type.update(label=new_label)
        return

    ## Embedding - TFIDF Hyperparams
    def dd_tfidf_strip_accents_callback(self, event):
        new_label = f"strip_accents={event.item}"
        self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_strip_accents.update(label=new_label)
        self.dashboard.update_selections(tfidf_strip_accents=event.item if event.item != "None" else None)
        return 

    def dd_tfidf_norm_callback(self, event):
        new_label = f"norm={event.item}"
        self.dashboard.nlp_tab.embedding_scatter.dd_tfidf_norm.update(label=new_label)
        self.dashboard.update_selections(tfidf_norm=event.item if event.item != "None" else None)
        return 

    def slider_tfidf_max_df_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_max_df_proportion=new)
        return 

    def slider_tfidf_min_df_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_min_df_proportion=new)
        return 

    def spinner_tfidf_max_df_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_max_df_count=new)
        return 

    def spinner_tfidf_min_df_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_min_df_count=new)
        return 

    def switch_tfidf_lowercase_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_lowercase=new)
        return 

    def switch_tfidf_binary_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_binary=new)
        return 

    def switch_tfidf_use_idf_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_use_idf=new)
        return 

    def switch_tfidf_smooth_idf_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_smooth_idf=new)
        return 

    def switch_tfidf_sublinear_tf_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_sublinear_tf=new)
        return 

    def radio_tfidf_min_df_callback(self, attr, old, new):
        self.dashboard.update_selections(tfidf_specify_min_df_as=self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_min_df.labels[new])
        
        if self.dashboard.selections['tfidf_specify_min_df_as'] == "proportion":
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.update(visible=False)
        elif self.dashboard.selections['tfidf_specify_min_df_as'] == "count":
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_min_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_min_df.update(visible=True)

        return

    def radio_tfidf_max_df_callback(self, attr, old, new):

        self.dashboard.update_selections(tfidf_specify_max_df_as=self.dashboard.nlp_tab.embedding_scatter.radio_tfidf_max_df.labels[new])
        
        if self.dashboard.selections['tfidf_specify_max_df_as'] == "proportion":
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.update(visible=False)
        elif self.dashboard.selections['tfidf_specify_max_df_as'] == "count":
            self.dashboard.nlp_tab.embedding_scatter.slider_tfidf_max_df.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.spinner_tfidf_max_df.update(visible=True)

        return

    ##### Dimensionality Reduction #####
    def dropdown_dimreduce_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dropdown_dimreduce.update(label=f"Dimensionality Reduction = {event.item}")
        #self.dashboard.selections["dimreduce_type"] = event.item
        self.dashboard.update_selections(dimreduce_type=event.item)

        if len(event.item.split("__")) > 1:
            self.dashboard.nlp_tab.embedding_scatter.slider_step1_dimreduce_ndims.update(visible=True)
        else:
             self.dashboard.nlp_tab.embedding_scatter.slider_step1_dimreduce_ndims.update(visible=False)

        if event.item == "umap":
            self.dashboard.nlp_tab.embedding_scatter.dd_umap_metric.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.slider_umap_n_neighbors.update(visible=True)
        else:
            self.dashboard.nlp_tab.embedding_scatter.dd_umap_metric.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.slider_umap_n_neighbors.update(visible=False)
                
        if "tsne" in event.item:
            self.dashboard.nlp_tab.embedding_scatter.slider_tsne_perplexity.update(visible=True)
        else:
            self.dashboard.nlp_tab.embedding_scatter.slider_tsne_perplexity.update(visible=False)

        ## Updating cluster_space Menu based on dimreduce_type
        #
        # Menu for two step dimensionality reduction methods
        if len(self.dashboard.selections['dimreduce_type'].split("__")) > 1:
            menu = [("Auto","auto"), 
                    ("Before Dim Reduce","before_dim_reduce"), 
                    ("After Dim Reduce Step 1", "after_dim_reduce_step1"),
                    ("After Full Dim Reduce", "after_full_dim_reduce")]

        else: # Remove "After Dim Reduce Step 1" from menu if method is one step only
            menu = [("Auto","auto"), 
                    ("Before Dim Reduce","before_dim_reduce"), 
                    ("After Full Dim Reduce", "after_full_dim_reduce")]

            # If we just took away the after_dim_reduce_step1 option and they were using it, 
            # reset there cluster space selection to auto
            if self.dashboard.selections['cluster_space'] == "after_dim_reduce_step1":
                self.dashboard.update_selections(cluster_space="auto")
                self.dashboard.nlp_tab.embedding_scatter.dd_cluster_space.update(label=f"Cluster Space = auto")

        self.dashboard.nlp_tab.embedding_scatter.dd_cluster_space.update(menu=menu)


        return 

    def slider_step1_dimreduce_ndims_callback(self, attr, old, new):
        self.dashboard.update_selections(step1_dimreduce_ndims=new)
        return

    def dd_scale_before_dim_reduce_callback(self, event):
        new_label = f"Scaling Before Dim Reduce = {event.item}"
        self.dashboard.nlp_tab.embedding_scatter.dd_scale_before_dim_reduce.update(label=new_label)
        #self.dashboard.selections["scale_before_dim_reduce"] = event.item
        self.dashboard.update_selections(scale_before_dim_reduce=event.item)
        return

    def dropdown_umap_metric_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_umap_metric.update(label=f"umap metric = {event.item}")
        self.dashboard.update_selections(umap_metric=event.item)
        return

    def slider_tsne_perplexity_callback(self, attr, old, new):
        self.dashboard.update_selections(tsne_perplexity=new)
        #self.dashboard.selections['tsne_perplexity'] = new
        return

    def umap_neighbors_slider_callback(self, attr, old, new):
        self.dashboard.update_selections(umap_n_neighbors=new)
        #self.dashboard.selections['umap_n_neighbors'] = new
        return 

    ##### Clustering #####
    def dropdown_cluster_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dropdown_cluster.update(label=f"cluster = {event.item}")
        self.dashboard.update_selections(cluster=event.item)
            
        if self.dashboard.selections["cluster"] == "agglomerative":
            self.dashboard.nlp_tab.embedding_scatter.dd_agg_affinity.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.dd_agg_linkage.update(visible=True)
        else:
            self.dashboard.nlp_tab.embedding_scatter.dd_agg_affinity.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.dd_agg_linkage.update(visible=False)
                
        if self.dashboard.selections["cluster"] == "kmeans":
            self.dashboard.nlp_tab.embedding_scatter.slider_kmeans_ninit.update(visible=True)
            self.dashboard.nlp_tab.embedding_scatter.dd_kmeans_algo.update(visible=True)
        else:
            self.dashboard.nlp_tab.embedding_scatter.slider_kmeans_ninit.update(visible=False)
            self.dashboard.nlp_tab.embedding_scatter.dd_kmeans_algo.update(visible=False)

        return 

    def dd_cluster_space_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_cluster_space.update(label=f"Cluster Space = {event.item}")
        self.dashboard.update_selections(cluster_space=event.item)
        return

    def dd_scale_before_cluster_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_scale_before_cluster.update(label=f"Scaling Before Clustering = {event.item}")
        self.dashboard.update_selections(scale_before_cluster=event.item)
        return

    def cluster_slider_callback(self, attr, old, new):
        self.dashboard.update_selections(n_clusters= new)
        return

    def dd_agglomerative_affinity_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_agg_affinity.update(label=f"affinity = {event.item}")
        self.dashboard.update_selections(agg_cluster_affinity=event.item)
        return

    def dd_agglomerative_linkage_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_agg_linkage.update(label=f"linkage = {event.item}")
        self.dashboard.update_selections(agg_cluster_linkage = event.item)
        return

    def kmeans_ninit_callback(self, attr, old, new):
        self.dashboard.update_selections(kmeans_n_init = new)
        self.dashboard.nlp_tab.embedding_scatter.slider_n_clusters.update(label=f"KMeans Initializations = {new}")
        return 

    def dd_kmeans_algo_callback(self, event):
        self.dashboard.nlp_tab.embedding_scatter.dd_kmeans_algo.update(label=f"algorithm = {event.item}")
        self.dashboard.update_selections(kmeans_algorithm=event.item)
        return

    ### Metric bar plot column
    def bar_cluster_column_button_callback(self, attr, old, new):
        self.dashboard.update_selections(cluster_bar_column_plotted=self.dashboard.nlp_tab.cluster_barplot.cluster_bar_column_radio.labels[new])
        self.dashboard.nlp_tab.cluster_barplot.update_figure(df=self.dashboard.nlp_tab.text_viz_df, 
                                                             selections=self.dashboard.selections)
        return

    ### Metric bar plot metric
    def bar_cluster_metric_button_callback(self, attr, old, new):
        self.dashboard.update_selections(cluster_bar_metric_plotted=self.dashboard.nlp_tab.cluster_barplot.cluster_bar_metric_radio.labels[new])
        
        if self.dashboard.selections['cluster_bar_metric_plotted'] == "size":
            self.dashboard.nlp_tab.cluster_barplot.cluster_bar_column_radio.update(disabled=True)
        else:
            self.dashboard.nlp_tab.cluster_barplot.cluster_bar_column_radio.update(disabled=False)

        self.dashboard.nlp_tab.cluster_barplot.update_figure(df=self.dashboard.nlp_tab.text_viz_df, 
                                                             selections=self.dashboard.selections)
        return

    ##### Submit button #####
    def submit_button_callback(self, event):
        print(f"Updating self.dashboard in callback, current selections: {self.dashboard.selections}")
        # self.dashboard.update_selections(current_cluster_cmap_info=self.dashboard.nlp_tab.current_colors)
        self.dashboard.update_nlp_tab()
        return