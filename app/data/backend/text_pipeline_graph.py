import pygraphviz as pgv
from sklearn.pipeline import Pipeline

class TextPipelineGraph:
    def __init__(self, num_text_documents):
        
        self.num_text_documents = num_text_documents
        self.graph = None
        self.embedding_subgraph = None
        self.cluster_subgraph = None
        self.dim_reduce_subgraph = None

        self._setup_graph()

        return 
    
    def add_edges(self, cluster_dimensionality, scale_before_dim_reduce):

        self.connect_subgraph_nodes()

        self.graph.add_edge("start", "embedding")
        self.graph.add_edge("embedding", "reduce_1")
        self.graph.add_edge(self.dim_reduce_subgraph.nodes()[-1], "end")
        self.graph.add_edge(self.cluster_subgraph.nodes()[-1], "end")

        if cluster_dimensionality == "before_dim_reduce":
            self.graph.add_edge("embedding", self.cluster_subgraph.nodes()[0])
        elif cluster_dimensionality == "after_full_dim_reduce":
            self.graph.add_edge(self.dim_reduce_subgraph.nodes()[-1], self.cluster_subgraph.nodes()[0])
        elif cluster_dimensionality == "after_dim_reduce_step1" and scale_before_dim_reduce in ["min_max", "standard_scaler"]:
            self.graph.add_edge(self.dim_reduce_subgraph.nodes()[1], self.cluster_subgraph.nodes()[0])
        elif cluster_dimensionality == "after_dim_reduce_step1":
            self.graph.add_edge(self.dim_reduce_subgraph.nodes()[0], self.cluster_subgraph.nodes()[0])
        
        return 

    def connect_subgraph_nodes(self):
        
        subgraphs = self.graph.subgraphs()

        for subgraph in subgraphs:
            nodes = subgraph.nodes()
            if len(nodes) > 1:
                for index in range(len(nodes) - 1):
                    subgraph.add_edge(nodes[index], nodes[index+1])
        
        return 

    def add_embedding_node(self, document_vector_shape, embedding_type, hugging_face_model_checkpoint, 
                           hugging_face_embedding_type):

        if embedding_type == "hugging_face":
            label = (f"{hugging_face_model_checkpoint}\n" 
                     f"{hugging_face_embedding_type}\n"
                     f"Output Dims: {document_vector_shape}")
        else:
            label = (f"{embedding_type}\n" 
                     f"Output Dims: {document_vector_shape}")

        self.embedding_subgraph.add_node(n="embedding", 
                                         color="white", 
                                         shape="rect",
                                         label=label, 
                                         labeljust="c", 
                                         labelloc="t")
        return
    
    def add_cluster_nodes(self, scale_before_cluster, n_clusters, cluster_model):

        node_count = 1
        if scale_before_cluster == "min_max":
            self.cluster_subgraph.add_node(n=f"clustering_{node_count}", 
                                           color="white", 
                                           shape="rect", 
                                           label="Min-Max Scale\n(zero to one range)", 
                                           labeljust="c", 
                                           labelloc="t")
            node_count += 1

        elif scale_before_cluster == "standard_scaler":
            self.cluster_subgraph.add_node(n=f"clustering_{node_count}", 
                                           color="white", 
                                           shape="rect", 
                                           label="StandardScaler\n(zero mean, unit variance)", 
                                           labeljust="c", 
                                           labelloc="t")
            node_count += 1

        cluster_string = cluster_model.__str__().split("(")[0]
        label=f"{cluster_string}\n{n_clusters} clusters"
        self.cluster_subgraph.add_node(n=f"clustering_{node_count}", 
                                       color="white", 
                                       shape="rect", 
                                       label=label, 
                                       labeljust="c", 
                                       labelloc="t")                                   
        return 
    
    def add_dimreduce_nodes(self, scale_before_dim_reduce, reducer, partial_reducer, 
                            reduced_vectors, partial_reduced_vectors):

        node_count = 1
        if scale_before_dim_reduce == "min_max":

            self.dim_reduce_subgraph.add_node(n=f"reduce_{node_count}", 
                                              color="white", 
                                              shape="rect", 
                                              label="Min-Max Scale\n(zero to one range)", 
                                              labeljust="c", 
                                              labelloc="t")
            node_count += 1

        elif scale_before_dim_reduce == "standard_scaler":

            self.dim_reduce_subgraph.add_node(n=f"reduce_{node_count}", 
                                              color="white", 
                                              shape="rect", 
                                              label="StandardScaler\n(zero mean, unit variance)", 
                                              labeljust="c", 
                                              labelloc="t")
            node_count += 1
        
        if isinstance(reducer, Pipeline):
            step_one = partial_reducer.__str__().split("(")[0]
            label = f"{step_one}\nOutput Dims: {partial_reduced_vectors.shape}"
            self.dim_reduce_subgraph.add_node(n=f"reduce_{node_count}", 
                                              color="white", 
                                              shape="rect", 
                                              label=label, 
                                              labeljust="c", 
                                              labelloc="t")
            
            node_count += 1

            step_two = reducer.steps[-1][-1].__str__().split("(")[0]
            label = f"{step_two}\nOutput Dims: {reduced_vectors.shape}"
            self.dim_reduce_subgraph.add_node(n=f"reduce_{node_count}", 
                                              color="white", 
                                              shape="rect", 
                                              label=label, 
                                              labeljust="c", 
                                              labelloc="t")
            node_count += 1
        else:

            step_one = reducer.__str__().split("(")[0]
            label = f"{step_one}\nOutput Dims: {reduced_vectors.shape}"
            self.dim_reduce_subgraph.add_node(n=f"reduce_{node_count}", 
                                              color="white", 
                                              shape="rect", 
                                              label=label, 
                                              labeljust="c", 
                                              labelloc="t")
            node_count += 1

        return 
    
    def _setup_graph(self):
        
        self.graph = pgv.AGraph(strict=True, 
                                directed=True, 
                                size = "11.7,16.5",
                                name="Visualization Pipeline",
                                rankdir="LR",
                                splines="true",
                                overlap="scalexy") 
        
        
        self.graph.add_node(n="start",  
                            shape="diamond", 
                            style="rounded,filled",
                            color="#FF99F9",
                            label=f"Start\n{self.num_text_documents} text documents")
        
        self.graph.add_node(n="end",  
                            shape="diamond", 
                            style="rounded,filled", 
                            color="#ABA6FF",
                            label=f"End\n2D Visualization")
        
        self.embedding_subgraph = self.graph.add_subgraph(name="cluster_embedding", 
                                                          rank="same",
                                                          style="filled", 
                                                          #fillcolor="#79ADDC", 
                                                          fillcolor="#FF7BB5", 
                                                          label="Document Vectors")
        
        self.cluster_subgraph = self.graph.add_subgraph(name="cluster_clusters", 
                                                        rank="same",
                                                        style="filled", 
                                                        #fillcolor="#FFC09F", 
                                                        fillcolor="#8ABDFF",
                                                        label="Clustering")
        
        self.dim_reduce_subgraph = self.graph.add_subgraph(name="cluster_dimreduce", 
                                                           rank="same", 
                                                           style="filled", 
                                                           # fillcolor="#ADF7B6", 
                                                           fillcolor="#C57BFF", 
                                                           labeljust="l",
                                                           label="Dimensionality Reduction")
        
        return 