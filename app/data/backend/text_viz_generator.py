import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.base import TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from nltk.tokenize import word_tokenize
from .embedding_generator import EmbeddingGenerator
from .dimensionality_reduction import DimensionReducer
from .clustering import ClusterGenerator
from .text_pipeline_graph import TextPipelineGraph

class TextVizGenerator:
    def __init__(self, dataframe, text_column="Tweet Text", embedding_type="tfidf", fasttext_vector_type="trained_model", random_state=7742, 
                tfidf_stop_words=None, tfidf_max_df=1.0, tfidf_min_df=1, tfidf_max_features=None, 
                tfidf_norm="l2", tfidf_use_idf=True, tfidf_sublinear_tf=False, tfidf_binary=False,
                hugging_face_model_checkpoint="distilbert-base-uncased", hugging_face_embedding_type="start_token_hidden",
                tfidf_smooth_idf=True, tfidf_strip_accents=None, tfidf_lowercase=True, tfidf_analyzer="word", fasttext_wv_filepath=None, 
                fasttext_model_filepath="../03_models/fasttext_default_wiki_tweet.bin", dimreduce_n_components=50, final_dimensionality=2, 
                 tnse_perplexity=30.0, tsne_learning_rate="auto", tnse_n_iter=3_000, tsne_init="pca", dimreduce_type="linear__tsne",
                 tsne_method="exact", umap_n_neighbors=15, umap_metric="euclidean", scale_before_dim_reduce=None, 
                 umap_low_memory=False, n_jobs=32, verbose=True, n_clusters=5, cluster_type="agglomerative",
                 agg_cluster_affinity="cosine", agg_cluster_memory=None, agg_cluster_connectivity=None, 
                 agg_cluster_compute_full_tree=True, agg_cluster_linkage="average", agg_cluster_distance_threshold=None, 
                 agg_cluster_compute_distances=False, cluster_column=None, kmeans_init="k-means++", kmeans_n_init=10, 
                 kmeans_max_iter=300, kmeans_algorithm="lloyd", cluster_space="auto", scale_before_cluster=None):
        
        self.df = dataframe
        self.text_column = text_column 
        self.texts = self.df[self.text_column].to_numpy().astype(str)
        self.grapher = TextPipelineGraph(num_text_documents=len(self.texts))
        
        self.cluster_columns = self.text_column if cluster_column is None else cluster_column 
        self.cluster_on_embeddings = self.cluster_columns == self.text_column 

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.embedding_type = embedding_type 

        # EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(text=self.texts, 
                                                      embedding_type=embedding_type, 
                                                      fasttext_vector_type=fasttext_vector_type, 
                                                      random_state=random_state, 
                                                      n_jobs=n_jobs, 
                                                      tfidf_stop_words=tfidf_stop_words, 
                                                      tfidf_max_df=tfidf_max_df, 
                                                      tfidf_min_df=tfidf_min_df, 
                                                      tfidf_max_features=tfidf_max_features, 
                                                      tfidf_norm=tfidf_norm, 
                                                      tfidf_smooth_idf=tfidf_smooth_idf, 
                                                      tfidf_strip_accents=tfidf_strip_accents, 
                                                      tfidf_lowercase=tfidf_lowercase, 
                                                      tfidf_analyzer=tfidf_analyzer, 
                                                      tfidf_use_idf=tfidf_use_idf, 
                                                      tfidf_sublinear_tf=tfidf_sublinear_tf, 
                                                      tfidf_binary=tfidf_binary,
                                                      fasttext_wv_filepath=fasttext_wv_filepath, 
                                                      fasttext_model_filepath=fasttext_model_filepath, 
                                                      hugging_face_model_checkpoint=hugging_face_model_checkpoint, 
                                                      hugging_face_embedding_type=hugging_face_embedding_type)
        self.document_vectors = None 

        # DimensionReducer
        self.reducer = None
        self.reduced_vectors = None
        self.scale_before_dim_reduce = scale_before_dim_reduce

        # pca__tsne, svd__tsne, umap ... etc.
        self.dimreduce_type = dimreduce_type 
        self.reduction_steps = self.dimreduce_type.split("__")
        self.final_dimensionality = final_dimensionality     # 2 dims
        self.dimreduce_n_components = dimreduce_n_components # controls pca/svd 
        
        # Arguments for TSNE
        self.tnse_perplexity=tnse_perplexity
        self.tsne_learning_rate=tsne_learning_rate
        self.tnse_n_iter=tnse_n_iter
        self.tsne_init=tsne_init
        self.tsne_method=tsne_method

        self.umap_n_neighbors = umap_n_neighbors
        self.umap_metric = umap_metric
        self.umap_low_memory = umap_low_memory

        # Controls when clustering is performed
        self.cluster_space = cluster_space
        self.cluster_dimensionality = self._set_cluster_vector_space()
        self.vectors_to_cluster = None 
        
        self.silhouette = None 
        self.calinski_harabasz = None 
        self.davies_bouldin = None 
        self.dataframe_name = None 
        
        self.cluster = ClusterGenerator(n_clusters=n_clusters, 
                                        cluster_type=cluster_type, 
                                        scale_before_cluster=scale_before_cluster,
                                        agg_cluster_affinity=agg_cluster_affinity, 
                                        agg_cluster_memory=agg_cluster_memory, 
                                        agg_cluster_connectivity=agg_cluster_connectivity, 
                                        agg_cluster_compute_full_tree=agg_cluster_compute_full_tree, 
                                        agg_cluster_linkage=agg_cluster_linkage, 
                                        agg_cluster_distance_threshold=agg_cluster_distance_threshold, 
                                        agg_cluster_compute_distances=agg_cluster_compute_distances, 
                                        kmeans_init=kmeans_init, 
                                        kmeans_n_init=kmeans_n_init, 
                                        kmeans_algorithm=kmeans_algorithm, 
                                        kmeans_max_iter=kmeans_max_iter)

    def _create_pipeline_graph(self):

        self.grapher.add_embedding_node(document_vector_shape=self.document_vectors.shape, 
                                        embedding_type=self.embedding_type, 
                                        hugging_face_model_checkpoint=self.embedding_generator.hugging_face_model_checkpoint, 
                                        hugging_face_embedding_type=self.embedding_generator.hugging_face_embedding_type)
        
        self.grapher.add_dimreduce_nodes(scale_before_dim_reduce=self.scale_before_dim_reduce, 
                                         reducer=self.reducer.reducer, 
                                         partial_reducer=self.reducer.partial_reducer, 
                                         reduced_vectors=self.reduced_vectors, 
                                         partial_reduced_vectors=self.reducer.partial_reduced_vectors)

        self.grapher.add_cluster_nodes(scale_before_cluster=self.cluster.scale_before_cluster, 
                                       n_clusters=self.cluster.n_clusters, 
                                       cluster_model=self.cluster.cluster_model)


        #self.grapher.connect_subgraph_nodes()
        self.grapher.add_edges(cluster_dimensionality=self.cluster_dimensionality, 
                               scale_before_dim_reduce=self.scale_before_dim_reduce)

        self.grapher.graph.layout(prog="dot", args="-Gdpi=300")
        self.grapher.graph.draw(path="./pipeline_graph.png", format="png")

        return 

    def get_viz_dataframe(self):

        self._get_document_vectors()

        self._create_reduced_vectors()

        self.df["x_embed"] = self.reduced_vectors[:,0]
        self.df["y_embed"] = self.reduced_vectors[:,1]
        self.df['cluster'] = self._get_clusters()

        self._create_pipeline_graph()

        self.df.name = self.dataframe_name
        
        return self.df

    def _get_document_vectors(self):
        
        # Special case, so we only request vectors from openai when needed (if we don't already
        # have the ones we are requesting stored in our dataframe)
        if self.embedding_type.startswith("openai"):

            openai_embedding_column_name = self.embedding_type.split("__")[0]

            if openai_embedding_column_name in self.df.columns:
                
                # If we saved a pandas dataframe with a list in it, we use eval to turn that data back into lists (from strings),
                # but if we put a list in a pandas dataframe and haven't saved it back to .csv, it will still be a list
                # and we don't need to use eval
                vectors = self.df[openai_embedding_column_name].tolist()
                if isinstance(vectors[0], str):
                    self.document_vectors = [eval(vector) for vector in vectors]
                else:
                    self.document_vectors = vectors

                self.document_vectors = np.array(self.document_vectors)

            else:
                
                self.document_vectors = self.embedding_generator.get_document_vectors()

                self.df[openai_embedding_column_name] = self.document_vectors

                self.document_vectors = np.array(self.document_vectors)

        else:
            self.document_vectors = self.embedding_generator.get_document_vectors()

        return 

    def _set_cluster_vector_space(self):

        if self.cluster_space != "auto":
            return self.cluster_space
        
        ## Setting when to cluster for "auto" mode
        #
        # For two step dim reduce techniques, auto clusters after step one
        if self.dimreduce_type in ["linear__tsne", "pca__tsne", "svd__tsne"]:
            return "after_dim_reduce_step1"

        # For one step dim reduce techniques, cluster prior to any dim reduce
        elif self.dimreduce_type in ["umap", "tsne", "linear", "pca", "svd"]:
            return "before_dim_reduce"

    def _get_clusters(self):

        if self.cluster_dimensionality == "before_dim_reduce":
            self.vectors_to_cluster = self.document_vectors
            print(f"Clustering on the full document vectors of dimensionality {self.vectors_to_cluster.shape[1]}")
        elif self.cluster_dimensionality == "after_dim_reduce_step1":
            self.vectors_to_cluster =  self.reducer.partial_reduced_vectors
            print(f"Clustering after Dim Reduce Step 1, on vectors of dimensionality {self.vectors_to_cluster.shape[1]}")
        elif self.cluster_dimensionality == "after_full_dim_reduce":
            self.vectors_to_cluster =  self.reduced_vectors
            print(f"Clustering after Full Dim Reduce, on vectors of dimensionality {self.vectors_to_cluster.shape[1]}")

        if scipy.sparse.issparse(self.vectors_to_cluster):
            self.vectors_to_cluster = self.vectors_to_cluster.toarray()

        cluster_assignments = self.cluster.get_clusters(X=self.vectors_to_cluster)
        self.silhouette = silhouette_score(X=self.vectors_to_cluster, labels=cluster_assignments, metric="cosine")
        self.calinski_harabasz = calinski_harabasz_score(X=self.vectors_to_cluster, labels=cluster_assignments)
        self.davies_bouldin = davies_bouldin_score(X=self.vectors_to_cluster, labels=cluster_assignments)
        self.dataframe_name = f"silhouette: {self.silhouette:.5f}, calinski_harabasz: {self.calinski_harabasz:.5f}, davies_bouldin: {self.davies_bouldin:.5f}"
        
        return cluster_assignments

    def predict(self, X):

        X_pred_embed = self.embedding_generator.transform(X)

        if self.cluster_dimensionality == "before_dim_reduce":
            preds_to_cluster = X_pred_embed
        elif self.cluster_dimensionality == "after_dim_reduce_step1":
            preds_to_cluster = self.reducer.partial_reducer.transform(X_pred_embed)
        elif self.cluster_dimensionality == "after_full_dim_reduce":
            preds_to_cluster = self.reducer.reducer.transform(X_pred_embed)

        cluster_model = clone(self.cluster.cluster_model)

        cluster_data = np.concatenate((self.vectors_to_cluster, 
                                       np.array(preds_to_cluster).reshape(1, -1)), 
                                       axis=0)
        
        predicted_clusters = cluster_model.fit_predict(cluster_data)[-1]

        return predicted_clusters

    def _create_reduced_vectors(self):

        self.reducer = DimensionReducer(vectors=self.document_vectors, 
                                        dimreduce_type=self.dimreduce_type, 
                                        dimreduce_n_components=self.dimreduce_n_components, 
                                        final_dimensionality=self.final_dimensionality, 
                                        tnse_perplexity=self.tnse_perplexity, 
                                        tsne_learning_rate=self.tsne_learning_rate, 
                                        tnse_n_iter=self.tnse_n_iter, 
                                        tsne_init=self.tsne_init, 
                                        tsne_method=self.tsne_method, 
                                        umap_n_neighbors=self.umap_n_neighbors, 
                                        umap_metric=self.umap_metric,
                                        umap_low_memory=self.umap_low_memory,
                                        random_state=self.random_state, 
                                        n_jobs=self.n_jobs, 
                                        verbose=self.verbose, 
                                        reduction_steps=self.reduction_steps, 
                                        scale_before_dim_reduce=self.scale_before_dim_reduce)

        self.reduced_vectors = self.reducer.get_reduced_vectors()

        return 