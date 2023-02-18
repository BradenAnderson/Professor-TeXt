import numpy as np
import pandas as pd
import scipy
import umap
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DimensionReducer:
    def __init__(self, vectors, dimreduce_type="linear__tsne", dimreduce_n_components=50, final_dimensionality=2, 
                 tnse_perplexity=30.0, tsne_learning_rate="auto", tnse_n_iter=3_000, tsne_init="pca", 
                 tsne_method="exact", umap_n_neighbors=15, umap_metric="hellinger", scale_before_dim_reduce=None,
                 umap_low_memory=False,random_state=7742, n_jobs=32, verbose=True, reduction_steps=None,):
        
        # Array of document vectors to reduce dimensionality of 
        self.vectors = vectors
        self.random_state = random_state
        self.n_jobs=n_jobs
        self.input_is_sparse = isinstance(self.vectors, scipy.sparse._csr.csr_matrix)

        self.reducer = None
        self.partial_reducer = None
        self.partial_reduced_vectors = None
        self.verbose = verbose

        # For scaling document vectors prior to dimensionality reduction
        self.scale_before_dim_reduce = scale_before_dim_reduce

        # pca__tsne, svd__tsne, umap ... etc.
        self.dimreduce_type = dimreduce_type 
        self.reduction_steps = reduction_steps if reduction_steps is not None else self.dimreduce_type.split("__")
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

    def _set_scale_inputs(self):

        if self.scale_before_dim_reduce and scipy.sparse.issparse(self.vectors):
            msg = ("Warning: You selected to scale before dim reduce but the embedding chosen " 
                   "resulted in sparse document vectors. Embeddings are being converted to dense "
                   "arrays prior to scaling.\n")
            print(msg)
            self.vectors = self.vectors.toarray()

        if self.scale_before_dim_reduce == "min_max":
            self.scaled_vectors = MinMaxScaler().fit_transform(self.vectors)
        elif self.scale_before_dim_reduce == "standard_scaler":
            self.scaled_vectors = StandardScaler().fit_transform(self.vectors)
        else:
            self.scaled_vectors = self.vectors
        return 

    def get_reduced_vectors(self):
        
        # Set self.scaled_vectors
        self._set_scale_inputs()

        try: 
            self.reducer = self._get_dimensionality_reducer()
            reduced_vectors = self.reducer.fit_transform(self.scaled_vectors)

        except TypeError as e:

            if "tsne" in self.reduction_steps and self.input_is_sparse and self.tsne_init == "pca":
                print("WARNING: Cannot use tsne_init=pca with sparse input, using random initialization")
                self.tsne_init = "random"
            else:
                print(f"Error occured during _get_dimensionality_reducer: {e}")
            
            reduced_vectors = self.get_reduced_vectors()

        self._set_partially_reduced_vectors()

        return reduced_vectors

    def _set_partially_reduced_vectors(self):
        
        # Set self.scaled_vectors
        self._set_scale_inputs()

        # If we want to cluster in PCA space or SVD space before TNSE...
        self.partial_reducer = self._get_reduction_step(self.first_step, model_only=True)
        
        self.partial_reduced_vectors = self.partial_reducer.fit_transform(self.scaled_vectors)

        return 

    def _get_dimensionality_reducer(self):

        self.first_step = self.reduction_steps[0]

        if len(self.reduction_steps) > 1:
            steps = [self._get_reduction_step(reduction_step_name) 
                                    for reduction_step_name in self.reduction_steps]
            

            return Pipeline(steps=steps)
        
        else:

            return self._get_reduction_step(self.first_step, model_only=True)

    def _get_reduction_step(self, reduction_step_name, model_only=False):

        if reduction_step_name == "umap":
            return self._get_umap() if model_only else ("UMAP", self._get_umap())
        elif reduction_step_name == "tsne":
            return self._get_tsne() if model_only else ("TSNE", self._get_tsne())
        elif reduction_step_name == "linear":
            return self._get_linear() if model_only else ("linear", self._get_linear())
        elif reduction_step_name == "pca":
            return self._get_pca() if model_only else ("PCA", self._get_pca())
        elif reduction_step_name == "svd":
            return self._get_pca() if model_only else ("SVD", self._get_svd())
    
    def _get_umap(self):
        if self.verbose:
            print("Adding UMAP")
        return umap.UMAP(n_components=self.final_dimensionality, 
                         n_neighbors=self.umap_n_neighbors, # Lower=More Local Structure, Higher=Global structure but lose fine details
                         metric=self.umap_metric,           # euclidean, cosine
                         low_memory=self.umap_low_memory,
                         random_state=self.random_state)

    def _get_tsne(self):
        if self.verbose:
            print("Adding TSNE")
        return TSNE(n_components=self.final_dimensionality,
                        perplexity=self.tnse_perplexity, 
                        learning_rate=self.tsne_learning_rate, 
                        n_iter=self.tnse_n_iter, 
                        random_state=self.random_state, 
                        init=self.tsne_init,
                        method=self.tsne_method,
                        n_jobs=self.n_jobs)

    def _get_linear(self):
        if self.input_is_sparse:
            return self._get_svd()
        else:
            return self._get_pca()

    def _get_pca(self):
        if self.verbose:
            print("Adding PCA")
        return PCA(n_components=self.dimreduce_n_components if len(self.reduction_steps)>1 else self.final_dimensionality,
                   random_state=self.random_state)

    def _get_svd(self):
        if self.verbose:
            print("Adding SVD")
        return TruncatedSVD(n_components=self.dimreduce_n_components if len(self.reduction_steps)>1 else self.final_dimensionality,
                            random_state=self.random_state)