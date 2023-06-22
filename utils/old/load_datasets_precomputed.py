# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:37:42 2022

@author: User
"""

import os
import pickle
import numpy as np
import torch

def cross_validation_split(dataset, num_folds = 5, n_test_folds = 1, inverse_augmentation = True):
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
    break_point = 10000
    new_data = 0
    if inverse_augmentation:
            print("previous train size"+ str(len(train)))
            max_new_programs = 10
            for i, tup in enumerate(train):
                new_data += len(tup[1])
                if max_new_programs<i:
                    print(i)
                    break
                    
                start_index = 0
                temp_data = []
                temp_lab = []
                temp_segment_lab = []
                for index, lab in enumerate(tup[1]):
                    temp_segment_lab.append(lab)
                    if lab:
                        temp_data.append(tup[0][start_index:index+1])
                        start_index = index+1
                        temp_lab.append(temp_segment_lab)
                        temp_segment_lab = []
                print()
                combined = [torch.tensor([]),[]]
                for index in reversed(range(len(temp_data))):
                    combined[0] = torch.cat((combined[0], temp_data[index]), axis = 0)
                    combined[1].extend(temp_lab[index])
                train.append(combined)
            print("new train size"+str((len(train))))
    
    folds.append((train, test))
  return folds

def load_dataset_from_precomputed(embedding_directory,
                                  lab_file,
                                  delete_last_sentence = False, 
                                  compute_confidence_intervals = False,
                                  lab_from_array = False,
                                  inverse_augmentation = False):
    
    data = []
    if lab_from_array:
        labs = np.load(lab_file, allow_pickle = True)
    else:
        with open(lab_file, 'rb') as f:
            labs = pickle.load(f)
        assert isinstance(labs, dict)
        audio_dir = os.path.dirname(list(labs.keys())[0])
    
    root = embedding_directory
    
    for index, file in enumerate(os.listdir(embedding_directory)):
        embs = torch.from_numpy(np.load(os.path.join(root, file)).squeeze()) # squeezing in case an extra dimension made its way into the embedding collection (should be 2d)
        if lab_from_array:
            labs[index][-1] = 0
            data.append((embs, labs[index]))
        else:
            try:
                try:
                    file_name = audio_dir + '/' + file[:-4] + '.mp3'
                    if len(labs[file_name])<1:
                        print("Warning: {} has no data".format(file_name))
                        continue
                    labs[file_name][-1] = 0
                except KeyError:
                    file_name = audio_dir + '/' + file[:-4] + '.wav'
                    if len(labs[file_name])<1:
                        continue
                    labs[file_name][-1] = 0
                data.append((embs, labs[file_name]))
            except KeyError:
                try:
                    file_name = audio_dir + '/audio\\' + file[:-4] + '.mp3'
                    if len(labs[file_name])<1:
                        continue
                    labs[file_name][-1] = 0
                except KeyError:
                    file_name = audio_dir + '/BMAT-ATS\\' + file[:-4] + '.wav'
                    if len(labs[file_name])<1:
                        continue
                    labs[file_name][-1] = 0
            
                    
            # assert sum(labs[file_name])>0, "{} has no positive topic boundaries".format(file_name)
            if sum(labs[file_name])<1:
                print("Warning: {} has no positive topic boundaries".format(file_name))
            data.append((embs, labs[file_name]))

    
    
    folds = cross_validation_split(data, inverse_augmentation = False)
        
    return folds