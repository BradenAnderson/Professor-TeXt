global paths 
paths = {"fasttext_model_path":"./data/models/fasttext_default_wiki_tweet.bin", 
         "data_path":"./App/data/data_aggregated_to_unique_texts_20230211_1940.csv"}

global current_selections
current_selections = {"run_with_sample_data":True,
                      "text_column":"Text", 
                      "primary_key_column":"observation_number",
                      "numeric_metric_columns":["Spend", "Impressions", "Clicks", 
                                                "Video Starts", "Reach", "Video Full Completion", 
                                                "Engagements", "Results"],
                      "embedding_scatter_hover_columns":["Text", "Spend", "Impressions", "Clicks", "Reach", "Video Starts",
                                                         "Platform Count","Ad Count", "Ad Group Count", "Campaign Count", 
                                                         "Result Type Count", "Objective Count", "observation_number", "cluster"],
                      "hover_text_size":"12px", 
                      "hover_window_width":"350px",
                      "regression_table_number_format":'0.00000', # Number of digits to format numbers to in the regression tables
                      "axis_range_buffer_pct":0.05,   # Draw this pct below and above largest values on x and y when setting axis ranges
                      "dimreduce_type":"linear__tsne", 
                      "step1_dimreduce_ndims":50,
                      "embedding_type":"tfidf", #fasttext_mean
                      "tfidf_strip_accents":None ,
                      "tfidf_norm":"l2", 
                      "tfidf_specify_max_df_as":"proportion", 
                      "tfidf_max_df_proportion":1.0, 
                      "tfidf_max_df_count":1_000, 
                      "tfidf_specify_min_df_as":"count", 
                      "tfidf_min_df_proportion":0.0001, 
                      "tfidf_min_df_count":1, 
                      "tfidf_lowercase":True, 
                      "tfidf_binary":False, 
                      "tfidf_use_idf":True, 
                      "tfidf_smooth_idf":True, 
                      "tfidf_sublinear_tf":False,
                      "umap_metric":"euclidean", 
                      "fasttext_vector_type":"trained_model", 
                      "cluster":"agglomerative", 
                      "n_clusters":4, 
                      "cluster_space":"auto",         # What vector space to cluster in
                      "scale_before_cluster":None,    # Method of scaling prior to dim reduction (e.g, min_max, standard_scaler)
                      "scale_before_dim_reduce":None, # Method of scaling prior to dim reduction (e.g, min_max, standard_scaler)    
                      "umap_n_neighbors":10,
                      "current_metric_plotted":None, 
                      "tsne_perplexity":30.0, 
                      "agg_cluster_affinity":"cosine", 
                      "agg_cluster_linkage":"average", 
                      "kmeans_n_init":20, 
                      "kmeans_algorithm":"lloyd", 
                      "cluster_bar_column_plotted":"Spend", 
                      "cluster_bar_metric_plotted":"mean", 
                      "hugging_face_model_checkpoint":"distilbert-base-uncased", 
                      "hugging_face_embedding_type":"start_token_hidden", 
                      "mlr_model_type":"separate_lines", # separate_lines, parallel_lines, slr
                      "mlr_x_column":"Spend", 
                      "mlr_categorical_column":"cluster", 
                      "mlr_y_column":"Impressions", 
                      "mlr_y_column_transformation":None,
                      "current_tab":"app_settings", 
                      "current_cluster_cmap_info":None, 
                      "embedding_scatter_chart_width":1200,
                      "embedding_scatter_chart_height":600, 
                      "regression_scatter_chart_width":1200,
                      "regression_scatter_chart_height":600, 
                      "regression_controls_width":325, 
                      "regression_controls_height":40, 
                      "mlr_X_predict":None, 
                      "mlr_cluster_predict":0, 
                      "mlr_ad_text_predict":None, 
                      "mlr_observations_to_remove":None, 
                      "qq_residuals_column":"externally_studentized_residuals",
                      "qq_reference_line_type":"45", 
                      "rvl_residuals_column":"externally_studentized_residuals", # Resid vs Leverage Plot
                      "rvp_residuals_column":"externally_studentized_residuals",  # Resid vs Predicted Plot
                      "rhist_residuals_column":"externally_studentized_residuals", # Residual Histogram
                      "rhist_residuals_num_bins":30, 
                      "text_gen_openai_models": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
                      "text_gen_success_metric":"Impressions", 
                      "text_gen_success_metric_bigger_is_better":True,
                      "text_gen_num_successful_tweet_examples":5,
                      "text_gen_twitter_username":"SMU", 
                      "text_gen_tweets_to_exclude":["retweets", "replies"],
                      "text_gen_num_recent_tweet_examples":5, 
                      "text_gen_temperature":0, # zero is less random, 2 is max? 
                      "text_gen_top_p":1, 
                      "text_gen_max_tokens":50, 
                      "text_gen_num_tweets_to_generate":1, 
                      "text_gen_model_name":"text-davinci-003", 
                      "text_gen_tweet_topic":None, 
                      "text_gen_tweet_tone":None, 
                      "text_gen_prompt":None, 
                      "text_gen_use_recent_tweets":True,
                      "text_gen_use_success_tweets":True, 
                      "text_gen_run_tweet_retriever":True, ## Only call twitter api if text_gen_twitter_username, text_gen_tweets_to_exclude or text_gen_num_recent_tweet_examples were changed
                      "sim_matrix_filter":True,  
                      "sim_matrix_filter_type":"random", # 'random', 'focused', 'exact'
                      "sim_matrix_filter_size":10, # int 
                      "sim_matrix_filter_focus_id":'1', # string 
                      "sim_matrix_filter_focus_type":'most_similar',  # 'most_similar', 'least_similar', 'random'
                      "sim_matrix_filter_exact_ids":None # List of string
                      }