import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from nltk.tokenize import word_tokenize

########################################################### HELPER CLASSES ###########################################################

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return np.asarray(X.todense())


class FastTextMeanEmbedding(object):

    def __init__(self, wv_map):
        
        self.wv_map = wv_map
        self.words_with_vectors = list(wv_map.keys())
        
        if len(wv_map.keys())>0:
            self.dim = self.wv_map[list(self.wv_map.keys())[0]].shape[0]
        else:
            self.dim=0
            
    def fit(self, X, y=None):
        return self 
    
    def _get_mean_vector(self, doc):
        
        if not isinstance(doc, list):
                doc = word_tokenize(doc)
        
        #print(f"doc is: {doc}")
        filtered_doc = [word for word in doc if word in self.words_with_vectors]
        #print(f"filtered_doc is: {filtered_doc}")
        wvs = [self.wv_map[w] for w in filtered_doc]
        mean_vec = np.mean(np.array(wvs), axis=0)
        #print(f"mean_vec is: {mean_vec}")
        return mean_vec
        
    def transform(self, X):

        mean_vectors = np.array([self._get_mean_vector(doc=doc) for doc in X])
       
        return mean_vectors
        
    def fit_transform(self, X):

        self.fit(X=X)

        return self.transform(X=X)


class FastTextTFIDFEmbedding(object):

    def __init__(self, wv_map):
        
        self.wv_map = wv_map
        self.words_with_vectors = list(wv_map.keys())

        self.idf_weight_map = None 
        self.max_idf = None

        if len(wv_map.keys())>0:
            self.dim = self.wv_map[list(self.wv_map.keys())[0]].shape[0]
        else:
            self.dim=0
            
    def fit(self, X, y=None):
        tfidf = TfidfVectorizer()
        tfidf.fit(X)

        self.max_idf = max(tfidf.idf_)

        self.word_to_idf_weight = {word:tfidf.idf_[idf_index] for word, idf_index in tfidf.vocabulary_.items()}
        return self 
    
    def _get_mean_tfidf_vector(self, doc):
        
        if not isinstance(doc, list):
                doc = word_tokenize(doc)
                
        filtered_doc = [word for word in doc if word in self.words_with_vectors]

        idf_weighted_wvs = [self.wv_map[w]*self.word_to_idf_weight.get(w, self.max_idf) 
                                for w in filtered_doc]


        # Mean of the tfidf weighted word vectors
        mean_tfidf_vec = np.mean(np.array(idf_weighted_wvs), axis=0)
        return mean_tfidf_vec
        
    def transform(self, X):

        mean_vectors = np.array([self._get_mean_tfidf_vector(doc=doc) for doc in X])
       
        return mean_vectors

    def fit_transform(self, X):

        self.fit(X=X)

        return self.transform(X=X)
########################################################### END HELPER CLASSES #########################################################

