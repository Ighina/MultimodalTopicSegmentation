import os
import sys
import json
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

from utils.wiki_loader_sentences import *
from utils.choiloader_sentences import *

def expand_label(labels,sentences):
  new_labels = [0 for i in range(len(sentences))]
  for i in labels:
    new_labels[i] = 1
  return new_labels

def cross_validation_split(dataset, num_folds = 5, n_test_folds = 1):
  unit_size = len(dataset)//num_folds
  test_size = len(dataset)//num_folds * n_test_folds
  folds = []
  for i in range(num_folds):
    test_start_idx = i*unit_size
    test_end_idx = i*unit_size + test_size
    test = dataset[test_start_idx:test_end_idx]
    if i == num_folds+1-n_test_folds:
        test += dataset[:test_size//n_test_folds]
        train = dataset[test_size//n_test_folds:-test_size//n_test_folds]
    else:
        train = dataset[:test_start_idx] + dataset[test_end_idx:]
    folds.append((train, test))
  return folds

def load_dataset(dataset, 
                 delete_last_sentence = False, 
                 compute_confidence_intervals = False):
                     
    if dataset == 'BBC':
        with open('data/BBC/train.json') as f:
            train = json.load(f)
            
        train_d = []
        
        for show in train['Transcripts']:
            sents_list = []
            labs = []
            
            for segment in show['Items']:
                sentences = nltk.sent_tokenize(segment)
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            train_d.append([sents_list, labs, show['Date']])
            
        train_data = []
        
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0])))
            
        with open('data/BBC/test.json') as f:
            test = json.load(f)
            
        test_d = []
        
        for show in test['Transcripts']:
            sents_list = []
            labs = []
            
            for segment in show['Items']:
                sentences = nltk.sent_tokenize(segment)
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            test_d.append([sents_list, labs, show['Date']])
        
        test_data = []
        
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0])))
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data)]
    
    elif dataset == 'BBCAudio':
        data = []
        data_path = './data/AudioBBC/modconhack_20210604/data'
        for root, directories, files in os.walk(data_path):
            assert len(files)>0
            for file in files:
                if file[-4:] == 'json':
                    with open(os.path.join(root, file), 'rb') as f:
                        print(file)
                        
                        test = json.load(f)
                            
                        test_d = []
                        
                        sents_list = []
                        labs = []  
                        
                        cut_index = 0
                        
                        for segment in test['data']["getProgrammeById"]['segments']:
                                sentences = nltk.sent_tokenize(segment['transcript'])
                                if delete_last_sentence:
                                    sentences = sentences[:-1]
                                sents_list.extend(sentences)
                                
                                labs.append(len(sents_list)-1)
                                
                        test_d.append([sents_list, labs, 0])
                            
                            
                        for w in test_d:
                          if w[0]:
                            data.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data)
        
        test = False
    
    elif dataset == 'CNN':
        data = []
        for i in range(1, 11):
          doc = read_wiki_file('data/CNN10/doc' + str(i) + '.txt', remove_preface_segment=False, 
                               high_granularity=False, return_as_sentences=True)
          
          sents = []
          labs = []
          for subs in doc[0]:
            if subs.startswith('===='):
              labs.append(index)
            else:
              
              sentences = nltk.sent_tokenize(subs)
              if delete_last_sentence:
                sentences = sentences[:-1]

              sents.extend(sentences)
              index = len(sents)-1
          labs.append(len(sents)-1)
          path = 'data/CNN10/doc' + str(i) + '.txt'
          data.append([sents, labs, path])
        
        data_list = []
        
        for w in data:
          if w[0]:
            data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list)
    
    elif dataset == 'wiki':
        data = WikipediaDataSet('data/wiki_test_50', folder=True, only_letters=False)
        data_list = []
        for w in data:
            if w[0]:
                if delete_last_sentence:
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0][:-1]):
                      if index not in w[1]:
                        new_w0.append(sent)
                      else:
                        new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    data_list.append((new_w0, expand_label(new_labs, new_w0)))
                else:
                
                    data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list)

    elif dataset == 'icsi':
        data = []
        
        segment_dir = 'data/icsi_mrda+hs_corpus_050512/segments'
        
        segment_files = os.listdir(segment_dir)
        
        file_dir = 'data/icsi_mrda+hs_corpus_050512/data'
        
        for root, direct, files in os.walk(file_dir):
            for file in files:
                if file[-4:]=='dadb':
                    continue
                
                try:
                    seg_file = [x for x in segment_files if re.search(file[:-6], x)][0]
                    
                    seg = []
                    
                    with open(os.path.join(segment_dir, seg_file)) as f:
                        for line in f:
                            seg.append(re.findall('\d+\.\d+', line)[0])
                
                except IndexError:
                    continue
                
                df = pd.read_csv(os.path.join(root,file), header = None)
                
                tmp = pd.DataFrame(df.iloc[:,0].str.split('_').tolist(), columns = ['id', 'start', 'end'])
                
                df = pd.concat([df, tmp], axis = 1)
                
                segment_index = 0
                
                labs = []
                
                starts = tmp['start'].tolist()
                delete_indeces = []
                deleted = 0
                for index, i in enumerate(starts):
                    if segment_index < len(seg):
                        if int(i)>float(seg[segment_index])*1000:
                            
                            if segment_index > 0:
                                if delete_last_sentence:
                                    try:
                                        labs[-2] = 1
                                    except:
                                        pass
                                    labs = labs[:-1]
                                    delete_indeces.append(index-deleted)
                                    deleted += 1
                                
                                else:
                                    labs[-1] = 1
                            
                            segment_index += 1
                            
                    labs.append(0)
                
                labs[-1] = 1
                if delete_last_sentence:
                    new_list = df[1].tolist()
                    for delete_index in delete_indeces:
                        new_list.pop(delete_index)
                    data.append((new_list, labs))
                else:
                    data.append((df[1].tolist(), labs))
        
        folds = cross_validation_split(data)


    else:
        data = ChoiDataset('data/choi')
        data_list = []
        for w in data:
          if w[0]:
            if delete_last_sentence:
                new_labs = []
                new_w0 = []
                for index, sent in enumerate(w[0][:-1]):
                  if index not in w[1]:
                    new_w0.append(sent)
                  else:
                    new_labs.append(len(new_w0)-1)
                new_labs.append(len(new_w0)-1)
                data_list.append((new_w0, expand_label(new_labs, new_w0)))
            else:
                
                data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list, num_folds=7, n_test_folds = 2)
        
    return folds