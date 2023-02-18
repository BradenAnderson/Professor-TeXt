from transformers import AutoTokenizer, AutoModel
from datasets import Dataset # Huggingface Datasets
import numpy as np 
import torch

# from transformers import logging
# logging.set_verbosity_error()

class HuggingFaceEmbedder:
    def __init__(self, text, embedding_type="start_token_hidden", model_checkpoint="distilbert-base-uncased", 
                 tokenizer_padding=True, tokenizer_truncation=True, batched=True, 
                 batch_size=None):
        
        self.text = text
        self.embedding_type = embedding_type
        self.batched = batched
        self.batch_size = batch_size # Default to 1000
        
        # Huggingface Dataset Object
        self.ds = Dataset.from_dict({"text":text})
        
        # Load the tokenizer
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.tokenizer_padding = tokenizer_padding
        self.tokenizer_truncation = tokenizer_truncation
        self.encoded_texts = None ## Texts after applying tokenizer
        
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.hidden_state = None 
        self.document_vectors = None

        # For Transforming 
        self.pred_ds = None
        self.pred_encoded_texts = None
        self.pred_hidden_states = None
        self.pred_document_vectors = None
    
    def transform(self, X):

        self.pred_ds = Dataset.from_dict({"text":X})

        self.pred_encoded_texts = self.pred_ds.map(self._tokenize_texts, 
                                                   batched=self.batched, 
                                                   batch_size=self.batch_size)

        # Set dataset format for the encoded texts to pytorch tensors
        self.pred_encoded_texts.set_format("torch", columns=["input_ids", "attention_mask", "text"])

        if self.embedding_type == "start_token_hidden":
            
            self.pred_hidden_states = self.pred_encoded_texts.map(self._extract_start_token_hidden_states, 
                                                                  batched=self.batched)
            
            self.pred_document_vectors = self.pred_hidden_states['hidden_state'].numpy()

        elif self.embedding_type == "mean":

            self.pred_hidden_states = self.pred_encoded_texts.map(self._extract_all_token_hidden_states, 
                                                                  batched=self.batched)
            
            self.pred_document_vectors = np.mean(self.pred_hidden_states['hidden_state'].numpy(), axis=1)

        return self.pred_document_vectors

    def get_document_vectors(self):
        
        self._encode_texts()
        
        if self.embedding_type == "start_token_hidden":
            return self._get_start_token_hidden_state_embeddings()
        elif self.embedding_type == "mean":
            return self._get_mean_embedded_hidden_states()
        
    def _get_mean_embedded_hidden_states(self):
        
        self.hidden_states = self.encoded_texts.map(self._extract_all_token_hidden_states, 
                                                    batched=self.batched)
        
        self.document_vectors = np.mean(self.hidden_states['hidden_state'].numpy(), axis=1)
        
        return self.document_vectors

    def _get_start_token_hidden_state_embeddings(self):
        
        self.hidden_states = self.encoded_texts.map(self._extract_start_token_hidden_states, 
                                                   batched=self.batched)
        
        self.document_vectors = self.hidden_states['hidden_state'].numpy()
        
        return self.document_vectors
        
        
    def _tokenize_texts(self, batch):
        return self.tokenizer(batch['text'], 
                              padding=self.tokenizer_padding, 
                              truncation=self.tokenizer_truncation)
    
    def _encode_texts(self):
        self.encoded_texts = self.ds.map(self._tokenize_texts, 
                                         batched=self.batched, 
                                         batch_size=self.batch_size)
        
        # Set dataset format for the encoded texts to pytorch tensors
        self.encoded_texts.set_format("torch", 
                                      columns=["input_ids", "attention_mask", "text"])
        return
    
    def _extract_all_token_hidden_states(self, batch):
        """Almost identical to _extract_start_token_hidden_states only difference
           is that this function does not index to only grab the start token hidden
           state [:, 0], it grabs them all"""
    
        # Grab the column names that the model needs [input_ids, attention_mask]
        # as inputs, and place that data on the GPU
        model_inputs = {input_name:input_data.to(self.device) for input_name, input_data
                        in batch.items() if input_name in self.tokenizer.model_input_names}
        
        # Run inputs through the model and grab the final hidden state for
        # each item in the batch
        with torch.no_grad(): # Disables gradient calc to speed things up, since we don't need gradients for prediction
            hidden_states = self.model(**model_inputs).last_hidden_state
            
        return {"hidden_state": hidden_states.cpu().numpy()}

    def _extract_start_token_hidden_states(self, batch):
        
        """Extracts the hidden state associated with the first 
           token in each text. This will be some kind of start token
           that is added by the tokenizer (e.g., [CLS])... it is common
           practice to use only the start token when extracting hidden states
           for use in a downstream classifier, but we could instead combine all hidden
           states (e.g., average them)"""
        
        # Grab the column names that the model needs [input_ids, attention_mask]
        # as inputs, and place that data on the GPU
        model_inputs = {input_name:input_data.to(self.device) for input_name, input_data
                        in batch.items() if input_name in self.tokenizer.model_input_names}
        
        # Run inputs through the model and grab the final hidden state for
        # each item in the batch
        with torch.no_grad(): # Disables gradient calc to speed things up, since we don't need gradients for prediction
            last_hidden_state = self.model(**model_inputs).last_hidden_state
            
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}