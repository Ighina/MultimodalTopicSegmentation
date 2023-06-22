# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:37:48 2021

@author: Iacopo
"""

import torch
from torch.utils.data import Dataset, DataLoader
from models.lightning_model import *
# import umap
# import nltk
from sklearn.decomposition import PCA
import numpy as np
# import re
# import os

class AudioPortionDataset(Dataset):
    def __init__(self, lines, tag_to_ix, encoder = 'x-vectors', CRF = True, 
                 truncate = True, truncate_value = 100, 
                 umap_project = False, umap_project_value = 100,
                 umap_class = None, second_input = None, domain_adapt = False):
        self.minus = 0 if CRF else 1
        self.embeddings = [line[0] for line in lines]
        self.tgt_dataset = [line[1] for line in lines]
        if second_input is not None:
            self.embeddings2 = [line[0] for line in second_input]
        else:
            self.embeddings2 = []
        self.truncate = truncate
        self.tv = truncate_value
        
        self.encoder_name = encoder #re-use an elsewhere defined encoder

        self.da = False
        if domain_adapt:
            self.da = True
            self.domain = []
            for line in lines:
                try:
                    isdigit = int(line[2][0]) # if first letter of file is a digit, then we are dealing with RadioNews dataset, else is NonNews
                    self.domain.append(1)
                except ValueError:
                    self.domain.append(0)
        else:
            self.domain = [None for line in lines]


        self.reducer = None
        
        if umap_project and umap_class is None:
            # self.reducer = umap.UMAP(n_components=umap_project_value)
            self.reducer = PCA(n_components=umap_project_value)
            train_emb = []
            lengths = []
            for emb in self.embeddings:
                lengths.append(len(emb))
                train_emb.append(emb.detach().cpu().numpy())
            
            train_emb = np.concatenate(train_emb)
            train_emb = self.reducer.fit_transform(train_emb)
            self.embeddings = []
            index = 0
            for length in lengths:
                self.embeddings.append(torch.tensor(train_emb[index:index+length]))
                index = length
        
        elif umap_project:
            for index in range(len(self.embeddings)):
                self.embeddings[index] = torch.tensor(umap_class.transform(self.embeddings[index].detach().cpu().numpy())) # terrible one liner to transform validation and test data with train-fitted umap class
    
    def __getitem__(self, index):
        if self.embeddings2:
            return {
            'id': torch.tensor(index),
            'target': self.tgt_dataset[index],
            'embeddings': self.embeddings[index],
            'embeddings2': self.embeddings2[index],
            'domain': self.domain[index]
        }
        return {
            'id': torch.tensor(index),
            'target': self.tgt_dataset[index],
            'embeddings': self.embeddings[index],
            'domain': self.domain[index]
        }
        
    def __len__(self):
        return len(self.embeddings)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, continuous=False, truncate = True, truncate_value = 100):
            if len(values[0].shape)<2:
              return torch.stack(values)
            else:
              if truncate:
                  max_length=truncate_value
              else:
                  max_length = max(v.size(0) for v in values)
              result = torch.zeros((len(values),max_length, values[0].shape[1]))
              for i, v in enumerate(values):
                  if truncate:
                      seq_len = min(truncate_value, len(v))
                  else:
                      seq_len = len(v) 
                  result[i, :seq_len] = v[:seq_len]
              return result

        def merge_tags(tags, truncate = True, truncate_value = 100):
          if truncate:
              max_length = truncate_value
          else:
              max_length = max(v.size(0) for v in tags)
          result = torch.zeros((len(tags),max_length)) - self.minus
          for i, v in enumerate(tags):
              if truncate:
                  seq_len = min(truncate_value, len(v))
              else:
                  seq_len = len(v)
              result[i, :seq_len] = v[:seq_len]
          return result
          
        
        id = torch.tensor([s['id'] for s in samples])
        src_tokens = merge([s['embeddings'] for s in samples], truncate = self.truncate, truncate_value = self.tv)
        if self.embeddings2:
            src_tokens2 = merge([s['embeddings2'] for s in samples], truncate = self.truncate, truncate_value = self.tv)
        else:
            src_tokens2 = None
        tgt_tokens = merge_tags([torch.tensor(s['target']) for s in samples], truncate = self.truncate, truncate_value = self.tv)
        if self.truncate:
            src_lengths = torch.LongTensor([min(self.tv, len(s["embeddings"])) for s in samples])
        else:
            src_lengths = torch.LongTensor([len(s['embeddings']) for s in samples])

        if self.da:
            domains = [s["domain"] for s in samples]
        else:
            domains = None
            

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'src_tokens2': src_tokens2,
            'domain': domains
        }

class AudioPortionDatasetInference(Dataset):
    def __init__(self, lines, encoder = 'x-vectors', CRF = True, 
                 truncate = False, truncate_value = 100, 
                 umap_project = False, umap_project_value = 100,
                 umap_class = None):
        self.minus = 0 if CRF else 1
        self.embeddings = lines
        self.truncate = truncate
        self.tv = truncate_value
        
        self.encoder_name = encoder #re-use an elsewhere defined encoder

        self.reducer = None
        
        if umap_project and umap_class is None:
            # self.reducer = umap.UMAP(n_components=umap_project_value)
            self.reducer = PCA(n_components=umap_project_value)
            train_emb = []
            lengths = []
            for emb in self.embeddings:
                lengths.append(len(emb))
                train_emb.append(emb.detach().cpu().numpy())
            
            train_emb = np.concatenate(train_emb)
            train_emb = self.reducer.fit_transform(train_emb)
            self.embeddings = []
            index = 0
            for length in lengths:
                self.embeddings.append(torch.tensor(train_emb[index:index+length]))
                index = length
        
        elif umap_project:
            for index in range(len(self.embeddings)):
                self.embeddings[index] = torch.tensor(umap_class.transform(self.embeddings[index].detach().cpu().numpy())) # terrible one liner to transform validation and test data with train-fitted umap class
    
    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'embeddings': self.embeddings[index]
        }
        
    def __len__(self):
        return len(self.embeddings)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, continuous=False, truncate = True, truncate_value = 100):
            if len(values[0].shape)<2:
              return torch.stack(values)
            else:
              if truncate:
                  max_length=truncate_value
              else:
                  max_length = max(v.size(0) for v in values)
              result = torch.zeros((len(values),max_length, values[0].shape[1]))
              for i, v in enumerate(values):
                  if truncate:
                      seq_len = min(truncate_value, len(v))
                  else:
                      seq_len = len(v) 
                  result[i, :seq_len] = v[:seq_len]
              return result
          
        
        id = torch.tensor([s['id'] for s in samples])
        src_tokens = merge([s['embeddings'] for s in samples], truncate = self.truncate, truncate_value = self.tv)
        if self.truncate:
            src_lengths = torch.LongTensor([self.tv for s in samples])
        else:
            src_lengths = torch.LongTensor([len(s['embeddings']) for s in samples])
            

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths
        }

# class Predictor:
#     def __init__(self, trained_model, sentence_encoder, tag2ix = None, remove_char = None):
#         self.model = trained_model
#         if isinstance(sentence_encoder, str):
#             self.se = SentenceTransformer(sentence_encoder)
#         else:
#             self.se = sentence_encoder
#         self.remove_char = remove_char
#         if tag2ix is None:
#             self.tags = {0:0, 1:1, '<START>':2, '<STOP>':3}
#         else:
#             self.tags = tag2ix
    
#     def online_sents_encoder(self, input_sentences):
#         embs = []
#         for sentence in input_sentences:
#             embs.append(self.se.encode(sentence))
        
#         return torch.tensor(embs).unsqueeze(0), torch.LongTensor([len(embs)])
    
#     def preprocess(self, doc, delete_special_char = True, 
#                    delete_punct = False, just_words = False):
#         sentences = nltk.sent_tokenize(doc)
        
#         input_sentences = []
        
#         for sentence in sentences:
#             if just_words:
#                 input_sentence = re.sub('[^a-z\s]', '', sentence).strip()
#             elif delete_punct:
#                 input_sentence = re.sub('[^\w\s]', '', sentence)
#             elif delete_special_char:
#                 if self.remove_char is None:
#                     input_sentence = re.sub('[\=#@{\|}~\[\]\^_]', '', sentence)
#                 else:
#                     input_sentence = re.sub(self.remove_char, '', sentence)
            
#             if input_sentence:
#                 input_sentences.append(input_sentence)
        
#         return input_sentences
    
#     def predict(self, docs, batch = False, 
#                 delete_special_char = True, 
#                 delete_punct = False, 
#                 just_words = False,
#                 pretokenized_sents = None,
#                 device = None,
#                 verbose = False):
        
#         if device is None:
#             if torch.cuda.is_available():
#                 device = 'cuda'
#             else: 
#                 device = 'cpu'
        
#         results = []
#         if batch:
            
            
            
#             batch_size = min(100, len(docs))
            
#             if verbose:
#                 print('Aggregating {} documents together to process them in batch...'.format(batch_size))
            
#             tokenized_sents = []
            
#             for index, doc in enumerate(docs):
#                 if pretokenized_sents:
#                     tokenized_sents.append(pretokenized_sents[index])
#                 else:
#                     tokenized_sents.append(self.preprocess(doc, 
#                                                            delete_special_char, 
#                                                            delete_punct, 
#                                                            just_words))
                
                
                
#             InferDataset = SentenceDataset(tokenized_sents, 
#                                            self.tags, 
#                                            encoder = self.se,
#                                            precompute=True,
#                                            infer = True)
            
#             dl = DataLoader(InferDataset, batch_size = batch_size, 
#                             collate_fn = InferDataset.collater)
            
#             results = []
            
#             for index, batch in enumerate(dl):
#                 if verbose:
#                     print('Segmenting batch number {}...'.format(index))
                
#                 inputs = batch['src_tokens'].to(device)
#                 lengths = batch['src_lengths'].to(device)
#                 input_sentences = batch['src_sentences']
                
#                 batch_scores, batch_boundaries = self.model.model(inputs, lengths)
                
#                 if verbose:
#                     print('Batch number {} segmented.'.format(index))
                
#                 for batch_index, boundaries in enumerate(batch_boundaries):
                    
#                     embs = inputs[batch_index][:lengths[batch_index]].squeeze().detach().cpu().numpy()
                    
                    
#                     segments = []
#                     segmented_embs = []
#                     last_index = 0
#                     for index, boundary in enumerate(boundaries):
#                         if boundary:
#                             segments.append(input_sentences[batch_index][last_index:index + 1])
#                             segmented_embs.append(embs[last_index:index + 1])
#                             last_index = index + 1
                    
#                     segments.append(input_sentences[batch_index][last_index:])
                    
#                     results.append({'segments': segments, 
#                         'boundaries': boundaries,
#                         'scores': batch_scores[batch_index],
#                         'embeddings': segmented_embs})
            
#         else:
#             for index, doc in enumerate(docs):
#                 if pretokenized_sents is None:
#                     input_sentences = self.preprocess(doc, 
#                                                        delete_special_char, 
#                                                        delete_punct, 
#                                                        just_words)
#                 else:
#                     input_sentences = pretokenized_sents[index]
                
#                 inputs, length = self.online_sents_encoder(input_sentences)
                
#                 scores, boundaries = self.model.model(inputs, length)
                
#                 embs = inputs.squeeze().numpy()
                
#                 segments = []
#                 segmented_embs = []
#                 last_index = 0
#                 for index, boundary in enumerate(boundaries[0]):
#                     if boundary:
#                         segments.append(input_sentences[last_index:index + 1])
#                         segmented_embs.append(embs[last_index:index + 1])
#                         last_index = index + 1
                
#                 segments.append(input_sentences[last_index:])
                
#                 results.append({'segments': segments, 
#                     'boundaries': boundaries,
#                     'scores': scores,
#                     'embeddings': segmented_embs})
        
#         return results
    
# def save_segmentation_results(results, output_directory, results_directory = 'results'):
#     save_dir = os.path.join(results_directory, output_directory)
    
#     assert not os.path.exists(save_dir)
    
#     os.makedirs(os.path.join(save_dir, 'segments'))
    
#     os.mkdir(os.path.join(save_dir, 'embeddings'))



# def get_model(model_name, max_length = None, labels = None):
    
#     if model_name.startswith('https') or model_name.lower().startswith('universal') or model_name.lower()=='use':
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
#         use = hub.load(module_url)
#         print ("module %s loaded" % module_url)
#         class SBERT_model():
#           def __init__(self, model):
#             self.model = model
#           def encode(self, sentences, convert_to_tensor = False):
#             if convert_to_tensor:
#                 return torch.from_numpy(self.model(sentences).numpy())
#             else:
#                 return self.model(sentences).numpy()
#           def get_sentence_embedding_dimension(self):
#             return 512
            
#         return (SBERT_model(use), None)
    
#     elif model_name.startswith('zero_'):
#         from transformers import pipeline
#         assert labels is not None, "To use the zero-shot approach provide possible topic labels"
#         model = model_name.split('_')[1]
#         device = 0 if torch.cuda.is_available() else -1
#         classifier = pipeline("zero-shot-classification", model=model, device = device)
#         class ZERO_model():
#             def __init__(self, model, labels):
#                 self.model=model
#                 self.labels=labels
                
#                 self.mapping = {k:v for v, k in enumerate(labels)}
#             def encode(self, sentences, convert_to_tensor = False):
#                 scores = []
                
#                 results = self.model(sentences, self.labels)
#                 for result in results:
#                     score = [0 for x in range(len(self.labels))]
#                     for index, lab in enumerate(result['labels']):
#                         score[self.mapping[lab]] = result['scores'][index]
#                     scores.append(score)
#                 if convert_to_tensor:
#                     return torch.tensor(scores)
#                 else:
#                     return np.array(scores)
            
#             def get_sentence_embedding_dimension(self):
#                 return len(self.labels)
            
#         return (ZERO_model(classifier, labels), None)   
    
#     elif model_name.startswith('ecapa'):
#         from speechbrain.pretrained import EncoderClassifier
#         model = EncoderClassifier.from_hparams(source = 'speechbrain/spkrec-ecapa-voxceleb',  savedir="audio_models/spkrec-ecapa-voxceleb")
#         class SBERT_model():
#           def __init__(self, model):
#             self.model = model
#           def encode(self, sentences, convert_to_tensor = False):
#             if convert_to_tensor:
#                 return torch.from_numpy(self.model.encode_batch(sentences).numpy())
#             else:
#                 return self.model.encode_batch(sentences).numpy()
#           def get_sentence_embedding_dimension(self):
#             return 192
#         return (SBERT_model(model), None)
    
#     elif model_name.startswith('xvectors'):
#         from speechbrain.pretrained import EncoderClassifier
#         model = EncoderClassifier.from_hparams(source = 'speechbrain/spkrec-xvectors-voxceleb',  savedir="audio_models/spkrec-xvectors-voxceleb")
#         class SBERT_model():
#           def __init__(self, model):
#             self.model = model
#           def encode(self, sentences, convert_to_tensor = False):
#             if convert_to_tensor:
#                 return torch.from_numpy(self.model.encode_batch(sentences).numpy())
#             else:
#                 return self.model.encode_batch(sentences).numpy()
#           def get_sentence_embedding_dimension(self):
#             return 512
#         return (SBERT_model(model), None)
    
#     elif model_name.startswith('bert-cls'):
        
#         from transformers import AutoTokenizer, AutoModel
        
#         class BERT_BASE_ENCODER:
#             def __init__(self):
#                     self.bert = AutoModel.from_pretrained('bert-base-uncased')
#                     self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#                     self.model = 'bert_cls_token'
#                     self.max_length = max_length
#                     if torch.cuda.is_available():
#                         self.device = 'cuda'
#                         self.bert.to('cuda')
#                         print('Moved BERT to gpu!')
#                     else:
#                         print('No gpu is being used')
#                         self.device = 'cpu'
        	        
#             def cls_pooling(self, model_output, attention_mask):
#      	        return model_output[0][:,0]
     	    
#             def encode(self, sentences, convert_to_tensor = False, batch_size = 32):
                
#                 all_embeddings = []
                
#                 length_sorted_idx = np.argsort([-len(sen.split()) for sen in sentences])
                
#                 sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
                
#                 for start_index in range(0, len(sentences), batch_size):
                    
#                     sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                    
#                     encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt', max_length = self.max_length)
                    
#                     with torch.no_grad():
                        
#                         if encoded_input['input_ids'].shape[0]>100:
#                             pass
                        
#                         model_output = self.bert(input_ids = encoded_input['input_ids'].to(self.device), attention_mask = encoded_input['attention_mask'].to(self.device))
                    
#                         model_output = self.cls_pooling(model_output, encoded_input['attention_mask']).detach().cpu()
                
#                         all_embeddings.extend(model_output)
                        
#                 all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
                
#                 if convert_to_tensor:
#                     return torch.stack(all_embeddings)
#                 elif convert_to_numpy:
#                     return np.asarray([emb.numpy() for emb in all_embeddings])
                    
#             def get_sentence_embedding_dimension(self):
#                 return 768
                
#         encoder = BERT_BASE_ENCODER()
#         return (encoder, None)
        
#     elif model_name.startswith('distil-bert-news'):
        
#         from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
#         class BERT_BASE_ENCODER:
#             def __init__(self):
  
#                 self.tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-agnews")
                
#                 self.bert = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-agnews")
                
#                 self.model = 'distil-bert-news'
                
#                 if torch.cuda.is_available():
#                     self.device = 'cuda'
#                     self.bert.to('cuda')
#                     print('Moved BERT to gpu!')
#                 else:
#                     print('No gpu is being used')
#                     self.device = 'cpu'
        	        
#             def cls_pooling(self, model_output, attention_mask):
#      	        return model_output[:,0]
     	    
#             def encode(self, sentences, convert_to_tensor = False):
                
#                 all_embeddings = []
                
#                 length_sorted_idx = np.argsort([-len(sen.split()) for sen in sentences])
                
#                 sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
                
#                 for start_index in range(0, len(sentences), batch_size):
                    
#                     sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                    
#                     encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt', max_length = self.max_length)
                    
#                     with torch.no_grad():
                        
#                         if encoded_input['input_ids'].shape[0]>100:
#                             pass
                        
#                         model_output = self.bert(input_ids = encoded_input['input_ids'].to(self.device), attention_mask = encoded_input['attention_mask'].to(self.device), output_hidden_states = True).hidden_states[-1]
                    
#                         model_output = self.cls_pooling(model_output, encoded_input['attention_mask']).detach().cpu()
                
#                         all_embeddings.extend(model_output)
                        
#                 all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
                
#                 if convert_to_tensor:
#                     return torch.stack(all_embeddings)
#                 elif convert_to_numpy:
#                     return np.asarray([emb.numpy() for emb in all_embeddings])
                    
#             def get_sentence_embedding_dimension(self):
#                 return 768
                
#         encoder = BERT_BASE_ENCODER()
#         return (encoder, None)
    
#     else:
#         if max_length is not None:
#             encoder = SentenceTransformer(model_name)
#             encoder.max_seq_length = max_length
            
#             return (encoder, None)
            
#         else:
#             return (SentenceTransformer(model_name), None)
    
    
    