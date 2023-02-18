import numpy as np
import pandas as pd
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from nltk.tokenize import word_tokenize

from .Helpers import FastTextMeanEmbedding, DenseTransformer, FastTextTFIDFEmbedding
from .hugging_face_embedder import HuggingFaceEmbedder
from .openai_embedder import OpenAIEmbedder

class EmbeddingGenerator:
    def __init__(self, text, embedding_type="tfidf", fasttext_vector_type="trained_model", random_state=7742, n_jobs=-1, 
                tfidf_stop_words=None, tfidf_max_df=1.0, tfidf_min_df=1, tfidf_max_features=None, tfidf_norm="l2", 
                tfidf_smooth_idf=True, tfidf_strip_accents=None, tfidf_lowercase=True, tfidf_analyzer="word", fasttext_wv_filepath=None, 
                tfidf_use_idf=True, tfidf_sublinear_tf=False, tfidf_binary=False,
                fasttext_model_filepath="../03_models/fasttext_default_wiki_tweet.bin", 
                hugging_face_model_checkpoint="distilbert-base-uncased", 
                hugging_face_embedding_type="start_token_hidden",):
        
        # Array of texts to embed
        self.text = text

        # Args to determine pipeline elements
        self.embedding_type = embedding_type             # tfidf, fasttext_mean, fasttext_tfidf
        self.fasttext_vector_type = fasttext_vector_type # trained_model, pretrained_vectors, allow for training ?? 

        # General args
        self.random_state = random_state
        self.n_jobs = n_jobs                  # currently only used by TSNE

        # Path to pretrained vectors downloaded from internet
        self.fasttext_wv_filepath = fasttext_wv_filepath

        # Path to model that we trained outselves
        self.fasttext_model_filepath = fasttext_model_filepath
        self.fasttext_model = None
        self.word_vector_map = None # Only for fasttext

        # Arguments for HuggingFaceEmbedder (when embedding_type="hugging_face")
        self.hugging_face_model_checkpoint = hugging_face_model_checkpoint
        self.hugging_face_embedding_type = hugging_face_embedding_type

        # Arguments for TFIDFVectorizer when embedding_type = "tfidf"
        self.tfidf_stop_words = tfidf_stop_words
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_norm = tfidf_norm
        self.tfidf_smooth_idf = tfidf_smooth_idf
        self.tfidf_analyzer = tfidf_analyzer
        self.tfidf_lowercase = tfidf_lowercase
        self.tfidf_strip_accents = tfidf_strip_accents
        self.tfidf_use_idf=tfidf_use_idf
        self.tfidf_sublinear_tf=tfidf_sublinear_tf
        self.tfidf_binary=tfidf_binary
        self.document_vectors = None

        self.tfidf = None 
        self.fasttext_mean_embedder = None 
        self.fasttext_tfidf_embedder = None 
        self.hugging_embedder = None
    
    def transform(self, X):
        
        if self.embedding_type == "tfidf":
            X_pred_embed = self.tfidf.transform(X)

        elif self.embedding_type == "fasttext_mean":
            X_pred_embed = self.fasttext_mean_embedder.transform(X)

        elif self.embedding_type == "fasttext_tfidf":
            X_pred_embed = self.fasttext_tfidf_embedder.transform(X)

        elif self.embedding_type == "hugging_face":
            X_pred_embed = self.hugging_embedder.transform(X)

        elif self.embedding_type.startswith("openai"):
            openai_model_name = self.embedding_type.split("__")[-1]
            oai_embedder = OpenAIEmbedder(text=X if isinstance(X, list) else [X], 
                                          model_name=openai_model_name)
            X_pred_embed = oai_embedder.get_document_vectors()
            X_pred_embed = np.array(X_pred_embed)
        
        return X_pred_embed

    def get_document_vectors(self):

        if self.embedding_type == "tfidf":
            self._get_tfidf_document_vectors()
        elif self.embedding_type == "fasttext_mean":
            self._get_fasttext_mean_document_vectors()
        elif self.embedding_type == "fasttext_tfidf":
            self._get_fasttext_tfidf_document_vectors()
        elif self.embedding_type == "hugging_face":
            self._get_hugging_transformer_document_vectors()
        elif self.embedding_type.startswith("openai"):
            self._get_openai_document_vectors()
        
        return self.document_vectors

    def _get_openai_document_vectors(self):
        
        openai_model_name = self.embedding_type.split("__")[-1]
        print(f"EMBEDDING TYPE: {self.embedding_type}")
        print(f"OPENAI MODEL NAME: {openai_model_name}")

        openai_embedder = OpenAIEmbedder(text=self.text.tolist(), model_name=openai_model_name)

        self.document_vectors = openai_embedder.get_document_vectors()

        return 

    def _get_hugging_transformer_document_vectors(self):

        self.hugging_embedder = HuggingFaceEmbedder(text=self.text, 
                                                    embedding_type=self.hugging_face_embedding_type, 
                                                    model_checkpoint=self.hugging_face_model_checkpoint)

        self.document_vectors = self.hugging_embedder.get_document_vectors()

        return 

    def _get_tfidf_document_vectors(self):

        self.tfidf = TfidfVectorizer(strip_accents=self.tfidf_strip_accents, 
                                     lowercase=self.tfidf_lowercase, 
                                     analyzer=self.tfidf_analyzer, 
                                     stop_words=self.tfidf_stop_words, 
                                     max_df=self.tfidf_max_df, 
                                     min_df=self.tfidf_min_df, 
                                     max_features=self.tfidf_max_features, 
                                     norm=self.tfidf_norm, 
                                     smooth_idf=self.tfidf_smooth_idf, 
                                     use_idf=self.tfidf_use_idf,
                                     sublinear_tf=self.tfidf_sublinear_tf, 
                                     binary=self.tfidf_binary)

        self.document_vectors = self.tfidf.fit_transform(self.text)
        
        return 

    def _get_fasttext_mean_document_vectors(self):

        self.fasttext_mean_embedder = self._get_fasttext_mean_embedder()

        self.document_vectors = self.fasttext_mean_embedder.fit_transform(np.char.lower(self.text))

        return

    def _get_fasttext_tfidf_document_vectors(self):

        self.fasttext_tfidf_embedder = self._get_fasttext_tfidf_embedder()

        self.document_vectors = self.fasttext_tfidf_embedder.fit_transform(np.char.lower(self.text))

        return

    def _get_fasttext_tfidf_embedder(self):

        if self.fasttext_vector_type == "trained_model":
            self._setup_for_fasttext_model()
        else: # setup to read in word vectors from file
            pass

        return FastTextTFIDFEmbedding(wv_map=self.word_vector_map)

    def _get_fasttext_mean_embedder(self):

        if self.fasttext_vector_type == "trained_model":
            self._setup_for_fasttext_model()
        else: # setup to read in word vectors from file
            pass

        return FastTextMeanEmbedding(wv_map=self.word_vector_map)
    
    def _setup_for_fasttext_model(self):
        self._load_fasttext_model()
        self._create_fasttext_word_vector_map()
        return

    def _load_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.fasttext_model_filepath)
        return

    def _create_fasttext_word_vector_map(self):

        all_tokens = [word_tokenize(txt.lower()) for txt in self.text]
        #print(f"Text is: {self.text}")

        # all_tokens = []
        # for txt in self.text:
        #     try:
        #         tokens = word_tokenize(txt.lower())
        #     except AttributeError as e:
        #         print(f"ERROR FOR TXT: {txt}\n\n {e}")
        #         raise e


        vocabulary = set([tok for tokens in all_tokens for tok in tokens])
        self.word_vector_map = {word:self.fasttext_model.get_word_vector(word) for word in vocabulary}

        return


    # def _get_fasttext_tfidf_embedding_from_pretrained(self):
        
    #     self.unique_tokens = self._get_unique_tokens()
    #     self.wv_map = self._create_word_to_vector_map()
    #     ft_tfidf_embed = FastTextTFIDFEmbedding(wv_map=self.wv_map)
    #     return [(self.embedding_step_name, ft_tfidf_embed)]

    # def _get_fasttext_mean_embedding_from_pretrained(self):
        
    #     self.unique_tokens = self._get_unique_tokens()
    #     self.wv_map = self._create_word_to_vector_map()
    #     ft_mean_embed = FastTextMeanEmbedding(wv_map=self.wv_map)
    #     return [(self.embedding_step_name, ft_mean_embed)]
        
    # def _get_unique_tokens(self):
    #     return set(word_tokenize(" ".join(self.dataframe[self.text_column].tolist())))

    # def _create_word_to_vector_map(self):
        
    #     pretrained_wvs = self._load_required_fasttext_vectors()
        
    #     wv_map = {}
    #     for word, vec in pretrained_wvs.items():
    #         wv_map[word] = np.array(list(vec))
        
    #     return wv_map

    # def _load_required_fasttext_vectors(self):
    #     fin = io.open(self.fasttext_wv_filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    #     n, d = map(int, fin.readline().split())
    #     data = {}
    
    #     for line in fin:
    #         tokens = line.rstrip().split(' ')
    #         if tokens[0] in self.unique_tokens:
    #             data[tokens[0]] = map(float, tokens[1:])
    #         else:
    #             continue
    #     return data

