import openai
import numpy as np
import pandas as pd
import os

class OpenAIEmbedder:
    def __init__(self, text, model_name="text-embedding-ada-002"):
        
        print(f"INSTANTIATING OPENAI EMBEDDER")
        self.openai = openai 
        self.openai.api_key = os.environ['OPENAI_API_KEY']
        
        self.text = text
        self.model_name = model_name
        
        self.response = None
        self.document_vectors = None
        
    def get_document_vectors(self):
        
        self.response = self.openai.Embedding.create(input=self.text, 
                                                     model=self.model_name)
        
        self.document_vectors = [self.response['data'][index]['embedding'] for index in range(len(self.response['data']))]
        
        return self.document_vectors

