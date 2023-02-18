
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class ClusterGenerator:
    def __init__(self, cluster_type="agglomerative", random_state=7742, n_jobs=32, verbose=True, n_clusters=5, 
                 agg_cluster_affinity="cosine", agg_cluster_memory=None, agg_cluster_connectivity=None, 
                 agg_cluster_compute_full_tree=True, agg_cluster_linkage="average", agg_cluster_distance_threshold=None, 
                 agg_cluster_compute_distances=False, kmeans_init="k-means++", kmeans_n_init=10, kmeans_max_iter=300, 
                 kmeans_algorithm="lloyd", scale_before_cluster=None):
        
        self.vectors = None

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.cluster_type = cluster_type 
        self.cluster_model = None 

        # Agglomerative Clustering Algo Hyperparams
        self.n_clusters = n_clusters
        self.agg_cluster_affinity = agg_cluster_affinity 
        self.agg_cluster_memory = agg_cluster_memory
        self.agg_cluster_connectivity = agg_cluster_connectivity 
        self.agg_cluster_compute_full_tree = agg_cluster_compute_full_tree
        self.agg_cluster_linkage = agg_cluster_linkage 
        self.agg_cluster_distance_threshold = agg_cluster_distance_threshold 
        self.agg_cluster_compute_distances = agg_cluster_compute_distances

        self.kmeans_init=kmeans_init
        self.kmeans_n_init=kmeans_n_init
        self.kmeans_max_iter=kmeans_max_iter
        self.kmeans_algorithm = kmeans_algorithm

        self.scale_before_cluster = scale_before_cluster

    def get_clusters(self, X):

        self.vectors = X

        # Scale self.vectors if required.
        self._set_scale_inputs()

        self.cluster_model = self._get_cluster_model()

        # self.scaled_vectors = self.vectors if self.scale_before_cluster is None
        return self.cluster_model.fit_predict(self.scaled_vectors)

    def _set_scale_inputs(self):

        if self.scale_before_cluster == "min_max":
            self.scaled_vectors = MinMaxScaler().fit_transform(self.vectors)
        elif self.scale_before_cluster == "standard_scaler":
            self.scaled_vectors = StandardScaler().fit_transform(self.vectors)
        else:
            self.scaled_vectors = self.vectors
        return 

    def _get_cluster_model(self):
        
        if self.cluster_type == "agglomerative":
            return AgglomerativeClustering(n_clusters=self.n_clusters, 
                                           metric=self.agg_cluster_affinity, 
                                           memory=self.agg_cluster_memory, 
                                           connectivity=self.agg_cluster_connectivity, 
                                           compute_full_tree=self.agg_cluster_compute_full_tree, 
                                           linkage=self.agg_cluster_linkage, 
                                           distance_threshold= self.agg_cluster_distance_threshold, 
                                           compute_distances=self.agg_cluster_compute_distances)
                                           
        elif self.cluster_type == "kmeans":
            return KMeans(n_clusters=self.n_clusters, 
                          init=self.kmeans_init, 
                          n_init=self.kmeans_n_init, 
                          algorithm=self.kmeans_algorithm)
