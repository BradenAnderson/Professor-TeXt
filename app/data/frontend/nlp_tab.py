import numpy as np
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import Button, TabPanel
from bokeh.layouts import column, row
from bokeh.themes import Theme
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, Plasma

from ..backend.text_viz_generator import TextVizGenerator
from .embedding_scatter import EmbeddingScatterPlot
from .metric_cluster_bar import MetricClusterBarPlot
from .cosine_similarity_heatmap import CosineSimilarityMatrix
from .text_pipeline_plot import TextPipelinePlot

class NLPTab:
    def __init__(self, df, paths, text_column, primary_key_column):
        
        self.paths = paths
        self.selections = None 
        self.df = df

        self.text_column = text_column
        self.primary_key_column = primary_key_column 

        self.text_generator = None
        self.text_viz_df = None
        self.current_palette = None
        self.current_colors = None

        # Bokeh figures
        self.embedding_scatter = None
        self.cluster_barplot = None 
        self.cosine_similarity_matrix = None
        self.pipeline_plot = None

        self.submit_button = Button(label="Submit Settings", button_type="primary")
        self.embedding_scatter_controls = None # embedding, dimreduce, cluster
        self.controls = None

        return

    def setup_plots(self):
        self.run_text_viz_generator()
        self.initialize_plots()
        return 

    def initialize_plots(self):

        self.embedding_scatter = EmbeddingScatterPlot(text_viz_df=self.text_viz_df, 
                                                      color_settings=self.current_colors, 
                                                      selections=self.selections)
        self.embedding_scatter.initialize_figure(selections=self.selections)

        self.cluster_barplot = MetricClusterBarPlot(text_viz_df=self.text_viz_df, 
                                                    color_settings=self.current_colors, 
                                                    selections=self.selections) 
        self.cluster_barplot.initialize_figure(selections=self.selections)

        # NOTE: the scaled_vectors attribute is the same as the vectors attribute if no scaling was performed.
        self.cosine_similarity_matrix = CosineSimilarityMatrix(vectors=self.text_generator.cluster.scaled_vectors, 
                                                               vector_ids=self.text_viz_df[self.primary_key_column].tolist(), 
                                                               id_to_cluster_map={str(txt_id):cluster for txt_id, cluster 
                                                                                    in zip(self.text_viz_df[self.primary_key_column].tolist(), 
                                                                                           self.text_viz_df['cluster'].tolist())},
                                                               vector_id_name=self.primary_key_column)
        
        self.cosine_similarity_matrix.initialize_figure(selections=self.selections)

        self.pipeline_plot = TextPipelinePlot()
        self.pipeline_plot.initialize_figure(graph=self.text_generator.grapher.graph)

        return 

    def update_plots(self):

        self.run_text_viz_generator()

        print(f"Updating plots with: {self.current_colors}")

        self.embedding_scatter.update_figure(df=self.text_viz_df,
                                             selections=self.selections,
                                             color_settings=self.current_colors)

        self.cluster_barplot.update_figure(df=self.text_viz_df, 
                                           selections=self.selections, 
                                           color_settings=self.current_colors)
        
        self.cosine_similarity_matrix.update_figure(selections=self.selections, 
                                                    vectors=self.text_generator.cluster.scaled_vectors, 
                                                    vector_ids=self.text_viz_df[self.primary_key_column].tolist(), 
                                                    vector_id_name=self.primary_key_column, 
                                                    id_to_cluster_map={str(txt_id):cluster for txt_id, cluster 
                                                                        in zip(self.text_viz_df[self.primary_key_column].tolist(), 
                                                                        self.text_viz_df['cluster'].tolist())})

        self.pipeline_plot.update_figure(graph=self.text_generator.grapher.graph)
        
        return


    def _set_tab_layout(self):

        self.embedding_scatter_controls = column(self.submit_button, 
                                                 self.embedding_scatter.get_controls_layout())

        row1 = row(self.embedding_scatter_controls, self.embedding_scatter.figure)



        row2 = row(column(self.cluster_barplot.cluster_bar_metric_radio, 
                          self.cluster_barplot.cluster_bar_column_radio, 
                          self.cluster_barplot.figure), 
                   self.cosine_similarity_matrix.get_figure_layout())


        row3 = row(self.pipeline_plot.figure)

        self.layout = TabPanel(child=column(row1, row2, row3), title="Text Visualization")

        return

    def run_text_viz_generator(self):

        #print(f"\n=====\nSELECTIONS:{self.selections}\n=====\n")

        self.text_generator = TextVizGenerator(dataframe=self.df, 
                                               text_column=self.text_column,
                                               fasttext_vector_type=self.selections['fasttext_vector_type'],
                                               embedding_type=self.selections['embedding_type'],
                                               tfidf_norm=self.selections['tfidf_norm'],
                                               tfidf_use_idf=self.selections['tfidf_use_idf'],
                                               tfidf_sublinear_tf=self.selections['tfidf_sublinear_tf'],
                                               tfidf_binary=self.selections['tfidf_binary'],
                                               tfidf_smooth_idf=self.selections['tfidf_smooth_idf'],
                                               tfidf_strip_accents=self.selections['tfidf_strip_accents'],
                                               tfidf_lowercase=self.selections['tfidf_lowercase'], 
                                               tfidf_max_df=self.selections['tfidf_max_df_proportion'] if self.selections['tfidf_specify_max_df_as']=="proportion" else self.selections['tfidf_max_df_count'],
                                               tfidf_min_df=self.selections['tfidf_min_df_proportion'] if self.selections['tfidf_specify_min_df_as']=="proportion" else self.selections['tfidf_min_df_count'],
                                               dimreduce_type=self.selections['dimreduce_type'], 
                                               umap_n_neighbors=self.selections["umap_n_neighbors"],
                                               umap_metric=self.selections["umap_metric"], 
                                               scale_before_dim_reduce=self.selections["scale_before_dim_reduce"],
                                               cluster_type=self.selections["cluster"], 
                                               n_clusters=self.selections["n_clusters"], 
                                               scale_before_cluster=self.selections["scale_before_cluster"],
                                               cluster_space=self.selections['cluster_space'],
                                               tnse_perplexity=self.selections["tsne_perplexity"], 
                                               agg_cluster_affinity=self.selections["agg_cluster_affinity"], 
                                               agg_cluster_linkage=self.selections["agg_cluster_linkage"], 
                                               kmeans_n_init=self.selections["kmeans_n_init"], 
                                               hugging_face_model_checkpoint=self.selections["hugging_face_model_checkpoint"], 
                                               hugging_face_embedding_type=self.selections["hugging_face_embedding_type"],
                                               kmeans_algorithm=self.selections["kmeans_algorithm"], 
                                               fasttext_model_filepath=self.paths['fasttext_model_path'], 
                                               dimreduce_n_components=self.selections['step1_dimreduce_ndims'] # Number of components for svd or pca in two step techniques
                                               )

        self.text_viz_df = self.text_generator.get_viz_dataframe()
        
        self._set_current_colors()

        self._store_new_embeddings()

        return 

    def _store_new_embeddings(self):

        ## if we saved new openai embeddings, copy them to the nlp tab dataframe so we don't
        ## request them again the next time we make a TextVizGenerator
        new_openai_columns = [c for c in self.text_viz_df.columns 
                                if c.startswith("openai") and c not in self.df.columns]
        if new_openai_columns:
            for c in new_openai_columns:
                self.df[c] = self.text_viz_df[c].to_numpy()
        
        # TODO: Save the updated self.df to new csv... can even copy the current name in self.paths

        return 

    def _set_current_colors(self):
        self.current_palette = Plasma.get(len(np.unique(self.text_viz_df['cluster'])), 
                                          Plasma[np.max(list(Plasma.keys()))])

        self.current_colors = {'palette':self.current_palette, 
                               'low':self.text_viz_df['cluster'].min(), 
                               'high':self.text_viz_df['cluster'].max()}

        return

    def get_layout(self):

        self._set_tab_layout()

        return self.layout 

    def set_selections(self, selections):
        self.selections = selections
        return