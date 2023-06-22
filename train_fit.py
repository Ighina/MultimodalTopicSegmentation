# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:53:24 2021

@author: Iacopo
"""

import os
import re
import itertools
import sys
import json
import shutil
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from speechbrain.pretrained import EncoderClassifier

from EncoderDataset import *
from utils.load_datasets_precomputed import *
from models.lightning_model import *

def main(args):
    
    verbose = args.verbose

    if args.infer:
        assert os.path.exists(args.experiment_name), 'If using pre-trained model to infer only, the given folder must exists and include the checkpoint subfolder with the trained model weights'
    else:
        assert not os.path.exists(args.experiment_name), 'The name of this experiment has already be used: please change experiment name or delete all the existent results from {} folder to use this name'.format(args.experiment_name)
    
        os.makedirs(args.experiment_name)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
        args.num_gpus = 0
    
    if args.dataset == 'BBC' or args.standard_split is not None:
        test = True
    else:
        test = False
    
    tag_to_ix = {0:0, 1:1, '<START>':2, '<STOP>':3}
    
    bs = args.batch_size
    
    folds = load_dataset_from_precomputed(args.embedding_folder,
                                          args.lab_folder,
                                          delete_last_sentence = args.delete_last_sentence,
                                          inverse_augmentation = True,
                                          k_folds = args.k_folds,
                                          mask_inner_sentences = args.mask_inner_sentences,
                                          mask_probability = args.mask_probability,
                                          split = args.standard_split,
                                          timing_info = args.timing_file)
    
    if args.architecture=='BiLSTMLateFusion':
        folds2 = load_dataset_from_precomputed(args.embedding_folder2,
                                                args.lab_folder,
                                          delete_last_sentence = args.delete_last_sentence,
                                          inverse_augmentation = True,
                                          k_folds = args.k_folds,
                                          mask_inner_sentences = args.mask_inner_sentences,
                                          mask_probability = args.mask_probability,
                                          split = args.standard_split)
        if args.both_datasets:
            folds2 = add_dataset(args, folds2, fold2=True)
    
    else:
        folds2 = [[None, None, None] for fold in folds]
    
    if args.both_datasets:
        folds = add_dataset(args, folds)

    val_folder = False
    if args.standard_split is not None:
        val_folder = True
    print(os.getcwd())
    # print(folds[0][1][0][2])
    os.chdir(args.experiment_name)
    
    loaders = []
    
    valid_percentage = args.valid_percentage
    
    encoder = args.encoder
    
    CRF = True if args.architecture.lower().endswith('crf') else False
        
    if verbose and CRF:
        print('Using architecture with conditional random field layer')
    
    all_results = {}
    all_scores = {}
    best_results_val = 1 if args.metric=="WD" or args.metric=="Pk" or not args.search_threshold else 0
    
    if args.architecture=="Transformer" or args.architecture=="BiLSTMRestrictedMHA":
        truncate = True # Change it to true if you use adaptive uniform segmentation  
        tv = 3600
    else:
        truncate = False
        tv = 100
    
    if args.architecture=="SwitchBiLSTM":
        domain_adapt = True
    else:
        domain_adapt = False

    for fold_idx, fold in enumerate(folds):
      
      valid_split = int(len(fold[0])*valid_percentage)
      
      if args.no_validation or val_folder:
          train_dataset = AudioPortionDataset(fold[0], tag_to_ix, encoder = encoder, CRF =CRF, 
                                              truncate = truncate, truncate_value = tv, umap_project = args.pca_reduce, umap_project_value = args.pca_value,
                                              second_input = folds2[fold_idx][0], domain_adapt=domain_adapt)
      else:    
          train_dataset = AudioPortionDataset(fold[0][:-valid_split], tag_to_ix, encoder = encoder, CRF =CRF, 
                                              truncate = truncate, truncate_value = tv, umap_project = args.pca_reduce, umap_project_value = args.pca_value,
                                              second_input = folds2[fold_idx][0], domain_adapt=domain_adapt)
      if not args.no_validation:
          if val_folder:
            valid_dataset = AudioPortionDataset(fold[2], tag_to_ix, encoder = encoder, CRF =CRF, 
                                              truncate = truncate, truncate_value = tv, umap_project = args.pca_reduce, umap_project_value = args.pca_value,
                                              umap_class = train_dataset.reducer, second_input = folds2[fold_idx][2], domain_adapt=domain_adapt)
          else:                                    
            valid_dataset = AudioPortionDataset(fold[0][-valid_split:], tag_to_ix, encoder = encoder, CRF =CRF, 
                                              truncate = truncate, truncate_value = tv, umap_project = args.pca_reduce, umap_project_value = args.pca_value,
                                              umap_class = train_dataset.reducer, second_input = folds2[fold_idx][0][-valid_split:], domain_adapt=domain_adapt)
      # truncate_test = True if args.embedding_folder.startswith("podcast") else False

      truncate_test = False

      test_dataset = AudioPortionDataset(fold[1], tag_to_ix, encoder = encoder, CRF =CRF, 
                                              truncate = truncate, truncate_value = tv, umap_project = args.pca_reduce, umap_project_value = args.pca_value,
                                              umap_class = train_dataset.reducer, second_input = folds2[fold_idx][1], domain_adapt=domain_adapt)
      
      train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
      if verbose:
        print('Train loader has: {} documents'.format(len(train_dataset)))
      if not args.no_validation:
          valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
          if verbose:
              print('Validation loader has: {} documents'.format(len(valid_dataset)))
      else:
          valid_loader = None
      test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = test_dataset.collater)
      if verbose:
        print('Test loader has: {} documents'.format(len(test_dataset)))
      loaders.append((train_loader, valid_loader, test_loader))
            
              
    seed_everything(args.seed, workers = True)
    
    search_space = {'hidden_units': [args.hidden_units], 'number_layers': [args.num_layers], 
                    'dropin': [args.dropout_in], 'dropout': [args.dropout_out]}
    
    best_results = {'F1':0, 'Pk':1, 'WD':1}
    if args.metric.lower()=='b':
        best_results['B'] = 0
    
    if args.hyperparameters_search:
        hyperparameters = []
        if len(args.hidden_units_search_space)>0:
            search_space['hidden_units'] = args.hidden_units_search_space
        hyperparameters.append(search_space['hidden_units'])
        
        if len(args.number_layers_search_space)>0:
            search_space['number_layers'] = args.number_layers_search_space
        hyperparameters.append(search_space['number_layers'])
        
        if len(args.dropout_in_search_space)>0:
            search_space['dropin'] = args.dropout_in_search_space
        hyperparameters.append(search_space['dropin'])
        
        if len(args.dropout_out_search_space)>0:
            search_space['dropout'] = args.dropout_out_search_space
        hyperparameters.append(search_space['dropout'])
        
        hyperparameters = list(itertools.product(*hyperparameters))
        
        results_grid_f1 = {layer:[] for layer in search_space['number_layers']}
        results_grid_pk = {layer:[] for layer in search_space['number_layers']}
        results_grid_wd = {layer:[] for layer in search_space['number_layers']}
        
        # results = {layer:[] for layer in args.number_layers_search_space}
        
    with open('logs', 'w') as f:
        f.write('Training started all right...\n')
    
    for param_index, param_tuple in enumerate(hyperparameters):
        
        results = []
        
        hu, nl, d_in, d_out = param_tuple
        
        if args.hyperparameters_search:
            with open('logs', 'a') as f:
                f.write('Results for model with {} hidden units, {} layers, {} dropout in, {} dropout out and {} batch size...\n'.format(hu, nl, d_in, d_out, bs))
        
        for index, segm in enumerate(loaders):
            if args.metric == 'Pk' or args.metric == 'WD' or not args.search_threshold:
            #if args.metric == 'Pk' or args.metric == 'WD':
                mode = 'min'
            else:
                mode = 'max'

            monitor = 'training_loss' if args.no_validation else 'val_loss'

            early_stop = EarlyStopping(
              monitor = monitor,
              patience = args.patience,
              strict = False,
              verbose = True,
              mode = mode)
            
            check_dir = 'checkpoints'
            
            if args.save_all_checkpoints:
                check_dir = check_dir + '_{}'.format(index)
            
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            elif not args.infer:
                os.remove(os.path.join(check_dir, os.listdir(check_dir)[-1]))
                
            # mode = 'max' if args.metric == 'F1' else 'min'
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                dirpath= check_dir,
                filename='checkpoint-{epoch:02d}-{val_loss:.4f}-{threshold:.2f}',
                save_top_k=1,
                mode=mode,
            )
            
            tagset_size = 2

            embedding_sizes = {'prosodic':167, 'openl3_std':1024, 'openl3/_mean_std':1024,
                                'wav2vec_std':1536, 'wav2vec/_mean_std':1536, 'x-vectors':512,
                                'openl3':512, 'crepe_std':512, 'crepe/mean_std':512,
                                'crepe':256, 'mfcc':200, 'ecapa':192, 'wav2vec':768,
                                'radio_news_topseg':768, 'non_news_topseg':768,
                                'radio_news_roberta':768, 'non_news_roberta': 768, 'CNN':30}

            if args.pca_reduce:
                embedding_dim = args.pca_value

            else:
                """
                The below code takes care of inferring all the number of encoders we want to use:
                for late fusion, different modalities are provided separately, for early fusion the encoders
                are provided all together in the same string.
                """
                try:
                    if folds2[0][0] is not None:
                        encoders = ["/".join(encoder.split("/")[1:]) for encoder in args.encoder.split("+")]
                        encoders2 = ["/".join(encoder.split("/")[1:]) for encoder in args.encoder2.split("+")]
                        embedding_dim = [sum([embedding_sizes[encoder] for encoder in encoders]), sum([embedding_sizes[encoder] for encoder in encoders2])]
                    else:
                        if re.findall("sentence", args.encoder.lower()):
                            encoders = ["/".join(encoder.split("/")[1:]) for encoder in args.encoder.split("+")]
                        else:
                            encoders = args.encoder.split("+")
                        embedding_dim = sum([embedding_sizes[encoder] for encoder in encoders])
                except KeyError:
                    raise ValueError("Encoder not recognised, use one of the available options (x-vectors, openl3, mfcc, prosodic, CREPE, ecapa or wav2vec)")
            
            if args.timing_file is not None:
                embedding_dim+=2

            train_loader, valid_loader, test_loader = segm

            if len(test_loader)==0:
                raise ValueError("There is something wrong with the test loader...")
                
            if args.no_early_stop:
                trainer = Trainer(callbacks = [checkpoint_callback], 
                            max_epochs = args.max_epochs, 
                            gpus = args.num_gpus, 
                            auto_lr_find = args.auto_lr_finder,
                            gradient_clip_val = args.gradient_clipping,
                            detect_anomaly = True)
            else:
                trainer = Trainer(callbacks = [early_stop, checkpoint_callback], 
                            max_epochs = args.max_epochs, 
                            gpus = args.num_gpus, 
                            auto_lr_find = args.auto_lr_finder,
                            gradient_clip_val = args.gradient_clipping,
                            detect_anomaly = True)

            if not args.infer:

                model = TextSegmenter(architecture = args.architecture,
                                    tagset_size = tagset_size, 
                                    embedding_dim = embedding_dim, 
                                    hidden_dim = hu, 
                                    lr = args.learning_rate, 
                                    num_layers = nl,
                                    LSTM = args.NoLSTM,
                                    bidirectional = args.unidirectional,
                                    loss_fn = args.loss_function,
                                    dropout_in = d_in,
                                    dropout_out = d_out,
                                    #dropout_out = 0,
                                    batch_first = args.batch_second,
                                    optimizer = args.optimizer,
                                    positional_encoding = args.positional_encoding,
                                    nheads = args.number_heads,
                                    search_threshold = args.search_threshold,
                                    metric = args.metric, no_validation = args.no_validation, 
                                    #alpha=d_out,
                                    attention_window = args.self_attention_window
                                    )

                if device == 'cuda':
                    model.cuda()

                if args.no_validation:
                    if args.auto_lr_finder:
                        trainer.tune(model, train_loader)
                    
                    trainer.fit(model, train_loader)
                    
                else:
                    if args.auto_lr_finder:
                        trainer.tune(model, train_loader, valid_loader)
                
                    trainer.fit(model, train_loader, valid_loader)
                
                threshold = args.threshold if args.threshold else float(checkpoint_callback.best_model_path.split('=')[-1][:4])
                best_val_loss = args.threshold if args.threshold else float(checkpoint_callback.best_model_path.split('=')[-2][:6]) 
                
                if args.no_validation or args.save_last_epoch:
                    trainer.save_checkpoint(os.path.join(check_dir,"final=0.500.ckpt"))
                    checkpoint_callback.best_model_path = os.path.join(check_dir,"final=0.500.ckpt")

            else:
                # TODO: include also option to do infer when args.save_last_epoch and args.no_validation are False
                checkpoint_callback.best_model_path = os.path.join(check_dir,"final=0.500.ckpt")
                threshold = 0.5      
            
            model = TextSegmenter.load_from_checkpoint(
                                  checkpoint_callback.best_model_path,
                                  architecture = args.architecture,
                                  tagset_size = tagset_size, 
                                  embedding_dim = embedding_dim, 
                                  hidden_dim = hu, 
                                  lr = args.learning_rate, 
                                  num_layers = nl,
                                  LSTM = args.NoLSTM,
                                  bidirectional = args.unidirectional,
                                  loss_fn = args.loss_function,
                                  dropout_in = 0.0,
                                  dropout_out = 0.0,
                                  batch_first = args.batch_second,
                                  optimizer = args.optimizer,
                                  positional_encoding = args.positional_encoding,
                                  nheads = args.number_heads,
                                  threshold = threshold,
                                  metric = args.metric,
                                  zero_baseline = args.zero_baseline,
                                  all_results = args.all_results,
                                  all_scores = args.all_scores,
                                  attention_window = args.self_attention_window)
            
            results.append(trainer.test(model, test_loader))
            
            if args.metric=='F1':
                f1_label = 'test_loss'
                pk_label = 'Pk_loss'
                wd_label = 'WD_loss'
            elif args.metric == 'WD':
                f1_label = 'F1_loss'
                pk_label = 'Pk_loss'
                wd_label = 'test_loss'
            elif args.metric.lower()=='b':
                f1_label = 'b_f1'
                pk_label = 'b_precision'
                wd_label = 'b_recall'
                test_label = 'test_loss'
            
            elif args.metric.lower()=='scaiano':
                f1_label = 'test_loss'
                pk_label = 'b_precision'
                wd_label = 'b_recall'
                
            else:
                f1_label = 'F1_loss'
                pk_label = 'test_loss'
                wd_label = 'WD_loss'
            
            if args.metric.lower() == 'b' or args.metric.lower() == 'scaiano':
                with open('logs', 'a') as f:
                    f.write('Results for fold number {}\n'.format(index))
                    f.write('B_precision score: {}\n'.format(results[-1][0][pk_label]))
                    f.write('B_recall score: {}\n'.format(results[-1][0][wd_label]))
                    f.write('B_F1 score: {}\n'.format(results[-1][0][f1_label]))
                    if args.metric.lower()=='b': 
                        f.write('B Similarity score: {}\n'.format(results[-1][0][test_label]))
            else:
                with open('logs', 'a') as f:
                    f.write('Results for fold number {}\n'.format(index))
                    f.write('PK score: {}\n'.format(results[-1][0][pk_label]))
                    f.write('WD score: {}\n'.format(results[-1][0][wd_label]))
                    f.write('F1 score: {}\n'.format(results[-1][0][f1_label]))
        
            if args.all_results:
            
                for result_index, file in enumerate(folds[index][1]):
                    # print(file[2])
                    all_results[file[2]] = model.results[result_index]
                    try:
                        val_loss = all_results[file[2]].pop('test_loss')
                        all_results[file[2]][args.metric] = val_loss
                    except:
                        pass
             
            if args.all_scores:
                for score_index, file in enumerate(folds[index][1]):
                    all_scores[file[2]] = model.scores[score_index].tolist()


        if test:
            f1 = results[-1][0][f1_label]
            pk = results[-1][0][pk_label]
            wd = results[-1][0][wd_label]
            if args.metric.lower()=='b':
                b = results[-1][0][test_label]
                
            if args.hyperparameters_search:     
                results_grid_f1[nl].append(f1)
                results_grid_pk[nl].append(pk)
                results_grid_wd[nl].append(wd)
            
            # if using new metrics, f1 map to f1, pk map to precision and wd map to recall (doing like this to avoid unnecessary code duplications)
            metrics = {'F1':f1, 'Pk': pk, 'WD': wd}
            if args.metric.lower()=='b':
                metrics['B'] = b

            #use_f1 = args.metric=='F1' or args.metric.lower()=='scaiano'
            
            #f1_best = (use_f1 and metrics['F1']>best_results['F1']) or (args.metric.lower() == 'b' and metrics['B']>best_results['B'])
            #f1_best = (use_f1 and best_val_loss>best_results_val) or (args.metric.lower() == 'b' and best_val_loss>best_results_val)

            if args.infer:
                best_results = metrics
                
                best_hu = hu
                best_nl = nl
                best_dropin = d_in
                best_dropout = d_out
                if args.all_results:
                    with open('all_results.json', 'w') as f:
                        json.dump(all_results, f)

                if args.all_scores:
                    with open('all_scores.json', 'w') as f:
                        json.dump(all_scores, f)
            else:
                f1_best = best_val_loss<best_results_val
            
                #if f1_best or (args.metric == 'Pk' and metrics['Pk']<best_results['Pk']) or (args.metric == 'WD' and metrics['WD']<best_results['WD']):
                if f1_best or (args.metric == 'Pk' and best_val_loss<best_results_val) or (args.metric == 'WD' and best_val_loss<best_results_val):
                    best_results = metrics
                    best_results_val = best_val_loss
                    
                    best_hu = hu
                    best_nl = nl
                    best_dropin = d_in
                    best_dropout = d_out
                    
                    if args.all_results:
                        with open('all_results.json', 'w') as f:
                            json.dump(all_results, f)

                    if args.all_scores:
                        with open('all_scores.json', 'w') as f:
                            json.dump(all_scores, f)
                    
                    try:
                        os.remove(os.path.join(check_dir, 'best_model'))
                    except:
                        pass
                    
                    new_name = os.path.join(check_dir, 'best_model')
                    
                    os.rename(checkpoint_callback.best_model_path, new_name)
                
        else:
            Pks = [p[0][pk_label] for p in results]
        
            F1s = [p[0][f1_label] for p in results]
                
            WDs = [p[0][wd_label] for p in results]
            if args.metric.lower()=='b':
                Bs = [p[0][test_label] for p in results]
                Avg_B = np.mean(Bs)
                
            Avg_PK = np.mean(Pks)
                
            Avg_F1 = np.mean(F1s)
                
            Avg_WD = np.mean(WDs)
            
            if args.hyperparameters_search:
                
                results_grid_f1[nl].append(Avg_F1)
                results_grid_pk[nl].append(Avg_PK)
                results_grid_wd[nl].append(Avg_WD)
                
            metrics = {'F1':Avg_F1, 'Pk': Avg_PK, 'WD': Avg_WD}
            if args.metric.lower()=='b':
                metrics['B'] = Avg_B
            
            #f1_best = (use_f1 and metrics['F1']>best_results['F1']) or (args.metric.lower() == 'b' and metrics['B']>best_results['B'])
            #f1_best = (use_f1 and best_val_loss>best_results_val) or (args.metric.lower() == 'b' and best_val_loss>best_results_val)

            f1_best = best_val_loss<best_results_val
            #if f1_best or (args.metric == 'Pk' and metrics['Pk']<best_results['Pk']) or (args.metric == 'WD' and metrics['WD']<best_results['WD']):
            if f1_best or (args.metric == 'Pk' and best_val_loss<best_results_val) or (args.metric == 'WD' and best_val_loss<best_results_val):
                best_results = metrics
                best_results_val = best_val_loss
                
                best_hu = hu
                best_nl = nl
                best_dropin = d_in
                best_dropout = d_out
                
                new_name = os.path.join(check_dir, 'best_model')
                
                os.rename(checkpoint_callback.best_model_path, new_name)
                
                def bootstrap(data, samples = 10000):
                  if isinstance(data, list):
                    data = pd.DataFrame(data)
                  boot = []
                  for sample in range(samples):
                    boot.append(data.sample(len(data), replace = True).mean()[0])
                  return boot
                
                boots = bootstrap(Pks)
                
                confidence_Pks = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                boots = bootstrap(F1s)
                
                confidence_F1s = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                boots = bootstrap(WDs)
                
                confidence_WDs = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                if args.metric.lower()=='b':
                    boots = bootstrap(Bs)
                    confidence_Bs = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
        
    if args.save_embeddings:
        if test:
            train_dataset.save_embeddings(args.encoder + '_train', args.dataset)
            valid_dataset.save_embeddings(args.encoder + '_valid', args.dataset)
            test_dataset.save_embeddings(args.encoder + '_test', args.dataset)
        else:
            train_dataset.embeddings = train_dataset.embeddings.extend(valid_dataset.embeddings)
            train_dataset.embeddings = train_dataset.embeddings.extend(test_dataset.embeddings)
            train_dataset.save_embeddings(args.encoder, args.dataset)
        
        args.save_embeddings = False
        
    if args.metric.lower()=='b' or args.metric.lower()=='scaiano':
        label_map = {'Pk':'Precision', 'WD': 'Recall', 'F1': 'F1'}
    else:
        label_map = {'Pk':'Pk', 'WD': 'WD', 'F1': 'F1'}

    if test:

        output = ['Results for experiment {} with following parameters:'.format(args.experiment_name),
              'Sentence encoder: {}'.format(args.encoder),
              'Neural architecture: {}'.format(args.architecture),
              'Batch size: {}'.format(args.batch_size),
              'Hidden units: {}'.format(best_hu),
              'Dropout in: {}'.format(best_dropin),
              'Dropout out: {}'.format(best_dropout),
              'Number of layers: {}'.format(best_nl),
              'Optimizer: {}'.format(args.optimizer),
              'Mean {} obtained is {}'.format(label_map['Pk'], best_results['Pk']),
              'Mean F1 obtained is {}'.format(best_results['F1']),
              'Mean {} obtained is {}'.format(label_map['WD'], best_results['WD'])]
        
        if args.metric.lower()=='b':
            output.append('Mean Boundary Similarity obtained is {}'.format(best_results['B']))
        
        if args.zero_shot_labels is not None:
            output.append('Labels: ' + str(args.zero_shot_labels))
        
    else:
        output = ['Results for experiment {} with following parameters:'.format(args.experiment_name),
                  'Sentence encoder: {}'.format(args.encoder),
                  'Neural architecture: {}'.format(args.architecture),
                  'Batch size: {}'.format(args.batch_size),
                  'Hidden units: {}'.format(best_hu),
                  'Dropout in: {}'.format(best_dropin),
                  'Dropout out: {}'.format(best_dropout),
                  'Number of layers: {}'.format(best_nl),
                  'Optimizer: {}'.format(args.optimizer),
                  'Mean {} obtained is {} with a 95% confidence interval of +- {}'.format(label_map['Pk'], best_results['Pk'], confidence_Pks),
                  'Mean F1 obtained is {} with a 95% confidence interval of +- {}'.format(best_results['F1'], confidence_F1s),
                  'Mean {} obtained is {} with a 95% confidence interval of +- {}'.format(label_map['WD'], best_results['WD'], confidence_WDs)]
        if args.metric.lower()=='b':
            output.append('Mean Boundary Similarity obtained is {} with a 95% confidence interval of +- {}'.format(best_results['B'], confidence_Bs))
        if args.zero_shot_labels is not None:
            output.append('Labels: ' + str(args.zero_shot_labels))
        
    if args.write_results:
            
        with open('results.txt', 'w') as f:
            for line in output:
                f.write('\n' + line + '\n')
        
    if args.hyperparameters_search:
        f1_results = pd.DataFrame(results_grid_f1)
        pk_results = pd.DataFrame(results_grid_pk)
        wd_results = pd.DataFrame(results_grid_wd)
        
        if args.write_results:
            f1_results.to_csv('F1_fit_results.csv')
            pk_results.to_csv('Pk_fit_results.csv')
            wd_results.to_csv('WD_fit_results.csv')
        
        return output, (f1_results, pk_results, wd_results)
    
    else:
        return output
    
if __name__ == '__main__':
    
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
                description = 'Run training with parameters defined in the relative json file')
    
    parser.add_argument('--experiment_name', '-exp', default = 'new_experiment', type = str,
                        help = 'The name of the current experiment (the output will be saved in a folder with the same name)')
    
    parser.add_argument('--dataset', '-data', default = 'choi', type=str,
                        help = 'The dataset to use in training. Options are choi, CNN or wiki')
    
    parser.add_argument('--batch_size', '-bs', default=64, type=int,
                        help = 'the size of each mini batch during training')
    
    parser.add_argument('--learning_rate', '-lr', default = 0.01, type = float,
                        help = 'The learning rate to be used during training')
    
    parser.add_argument('--valid_percentage', '-vp', default = 0.1, type = float,
                        help = 'The percentage of training data used for validation')
    
    parser.add_argument('--encoder', '-enc', default = 'stsb-bert-base', type = str,
                        help = 'The sentence encoder to be used to encode the sentences: possible options include all accepted values from sentence_transformers library and "use" for universal sentence encoder (DAN)')

    parser.add_argument('--encoder2', '-enc2', default = None, type = str,
                        help = 'The sentence encoder to be used to encode the sentences: possible options include all accepted values from sentence_transformers library and "use" for universal sentence encoder (DAN)')
    
    parser.add_argument('--online_encoding', '-oe', action = 'store_true',
                        help = 'If included, this option makes the dataloader encode each batch on the fly, instead of precomputing and accessing directly the stored embeddings (this option is useful if the embeddings do not fit in memory)')
    
    # parser.add_argument('--embeddings_directory', '-ed2', default = None, type = str,
    #                     help = 'The directory storing the precomputed embeddings. By default no directory is included and the embeddings are computed by the script directly. Precomputing the embeddings and storing them, however, result in massive saving of training time.')
    
    # parser.add_argument('--embeddings_directory2', '-ed', default = None, type = str,
    #                     help = 'The directory storing the precomputed embeddings. By default no directory is included and the embeddings are computed by the script directly. Precomputing the embeddings and storing them, however, result in massive saving of training time.')
    
    parser.add_argument('--patience', '-pat', default = 20, type = int, 
                        help = 'After how many bad iterations to stop training')
    
    parser.add_argument('--architecture', '-arc', default = 'biLSTMCRF', type = str,
                        help = 'Which neural architecture to use: implemented for now are BiLSTMCRF and Transformer-CRF.')
    
    parser.add_argument('--hidden_units', '-hu', default = 25, type = int,
                        help = 'How many hidden units to use')
    
    parser.add_argument('--num_layers', '-nl', default = 1, type = int,
                        help = 'How many layers to use')
    
    parser.add_argument('--NoLSTM', action = 'store_false',
                        help = 'If included, this option tells the network to use GRU instead of LSTM')
    
    parser.add_argument('--number_heads', '-nh', default = 8, type = int,
                        help = 'number of attention heads to be used in transformer.')
    
    parser.add_argument('--positional_encoding', '-pe', action = 'store_false',
                        help = 'if included, the option avoids using positional encoding in the transformers.')
                        
    parser.add_argument('--threshold', '-th', default = 0.0, type=float,
                        help = 'threshold to be used in inference for koshorek like models.')
    
    parser.add_argument('--unidirectional', action = 'store_false',
                        help = 'If included, this option tells the network to use a unidirectional RNN instead of bidirectional (default)')
    
    parser.add_argument('--max_length', type = int, required = False,
                        help = 'Just for Bert base and Bert news, the max input size of each sentence (if not included it default to 512 word pieces).')
    
    parser.add_argument('--dropout_in', '-d_in', default = 0.0, type = float,
                        help = 'The percentage of connections to randomly drop between embedding layer and first hidden layer during training')
    
    parser.add_argument('--dropout_out', '-d_out', default = 0.0, type = float,
                        help = 'The percentage of connections to randomly drop between last hidden layer and output layer during training')
    
    parser.add_argument('--batch_second', action = 'store_false',
                        help = 'If included, this option tells the network that the expected input has shape (seq_length, batch_size, embedding_size)')
    
    parser.add_argument('--optimizer', '-opt', default = 'Adam', type = str,
                        help = 'What optimizer to use: currently accepted are Adam or SGD (stochastic gradient descent)')
    
    parser.add_argument('--max_epochs', '-max', default = 100, type = int,
                        help = 'Number of training iterations after which to stop training (if early stopping mechanism does not stop the training before)')
    
    parser.add_argument('--num_gpus', '-gpus', default = 1, type = int,
                        help = 'Specify the number of gpus to use')
    
    parser.add_argument('--auto_lr_finder', '-auto_lr', action = 'store_true',
                        help = 'Include to activate the pytorch lightning functionality of finding the optimal learning rate before starting training')
    
    parser.add_argument('--save_all_checkpoints', '-savec', action = 'store_true',
                        help = 'If included, this option tells the program to save all the best checkpoints for each fold of the cross-validation process (extremely expensive memory wise)')
    
    parser.add_argument('--save_embeddings', '-savee', action = 'store_true',
                        help = 'If included, this option tells the script to save the embeddings extracted for the training, validation and test corpus. By default, these embeddings will be saved under embeddings/{encoder_name}_{one of train, validation or test}/embeddings_{sentence_number}')
    
    parser.add_argument('--use_end_boundary', '-ueb', action = 'store_true', help = 'Whether if to include the final sentence as a positive target in training (it will not be included when computing the metrics, but just for having additional positive classes in training)')
        
    parser.add_argument('--verbose', '-v', action = 'store_true',
                          help = 'Print out additional information during the training process')
    
    parser.add_argument('--write_results', '-wr', action='store_false',
                        help = 'If included, the results will not be written in results file.')
    
    parser.add_argument('--hyperparameters_search', '-hs', action = 'store_true',
                        help = 'If included, it will search for the best hidden units and layers numbers by doing a grid search among the options defined below and it will output a csv with all the results of the fitting process in addition to the standard results file.')
    
    parser.add_argument('--hidden_units_search_space', '-huss', nargs='*', required=False, type=int,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the hidden units values to be searched in the fitting process.')
    
    parser.add_argument('--number_layers_search_space', '-nlss', nargs='*', required=False, type=int,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the number of layers to be searched in the fitting process.')
    
    parser.add_argument('--dropout_in_search_space', '-diss', nargs='*', required=False, type=float,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the dropout value before inputting to the recurrent layer to be searched in the fitting process.')
    
    parser.add_argument('--dropout_out_search_space', '-doss', nargs='*', required=False, type=float,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the dropout value after the recurrent layer to be searched in the fitting process.')
    
    parser.add_argument('--batch_size_search_space', '-bass', nargs='*', required=False, type=int,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the batch sizes to be searched in the fitting process.')
    
    parser.add_argument('--metric', default = 'Pk', type = str, choices=['Pk', 'F1', 'WD', 'b', 'scaiano'],
                        help = 'The metric to use for determining the best hyperparameters in case the hyperparameters_search attribute is active (see above). Possible choices are Pk, F1 and WD (window difference)')
    
    parser.add_argument('--delete_last_sentence', '-dls', action = 'store_true',
                        help = 'This option deletes the last sentence from each segment in the input dataset so to test the robustness of the model under this condition.')
                        
    parser.add_argument('--zero_shot_labels', '-zsl', type = str, nargs='*',
                        help = 'If using zero shot approach (ZeroTiling) then provide the labels of the topic to be inferred')
                        
    parser.add_argument('--search_threshold', '-sth', action = 'store_true', help = 'Whether to search for the optimal threshold during training')
    
    parser.add_argument('--cosine_loss', '-cos', action = 'store_true', help = "Whether to include the cosine loss of the last layers' embeddings as an additional loss.")
    
    parser.add_argument('--gradient_clipping', '-gc', default = 0.0, type = float, help = 'The value to clip the gradient to.')
    
    parser.add_argument('--embedding_folder', '-ef', type=str, required = True,
                        help = 'The path to the directory storing the precomputed audio embeddings')

    parser.add_argument('--embedding_folder2', '-ef2', type=str, required = False, default = None,
                        help = 'The path to the directory storing the precomputed audio embeddings')
    
    parser.add_argument('--lab_folder', '-lf', type=str, required=True,
                        help = 'The path to the file storing the ground truth labels of each document')

    parser.add_argument('--inverse_augment', '-ia', action = 'store_true', help = 'whether to augment the dataset with the inverse of it.')
    
    parser.add_argument('--zero_baseline', '-zb', action = 'store_true', help = "test the baseline consisting in never predicting a boundary.")
    
    parser.add_argument('--loss_function', '-loss', choices = ["CrossEntropy", "BinaryCrossEntropy", "FocalLoss"], default = "CrossEntropy", help = "The loss function to be used during training")    

    parser.add_argument('--seed', default = 42, help = "The random seed to replicate the experiments")

    parser.add_argument('--no_validation', '-no_val', action = 'store_true', help = 'If this option is included, do not load checkpoint from best validation check')

    parser.add_argument('--no_early_stop', '-no_stop', action = 'store_true', help = 'If this option is included, do not perform early stopping')

    parser.add_argument('--save_last_epoch', '-s_last', action = 'store_true', help = 'Similar to no_validation option, but in this case instead of using all data for training it still perform validation but it saves the last epoch from training anyway.')

    parser.add_argument('--pca_reduce', '-pca', action = 'store_true', help = 'If included, use umap to reduce the embedding dimension to the dimension specified by --umap_value')

    parser.add_argument('--pca_value', '-pca_v', default = 167, type = int, help = 'The number of components for umap reducer (see above)')

    parser.add_argument('--all_results', '-ar', action = 'store_true', help = 'Whether to save all the individual test results instead of giving the aggregated scores.')

    parser.add_argument('--all_scores', '-as', action = 'store_true', help = 'Whether to save all the individual test scores (posterior probabilities).')

    parser.add_argument('--k_folds', '-kcv', default = 5, type = int, help = 'the value of k for the cross validation procedure.')

    parser.add_argument('--mask_inner_sentences', '-msk', action = 'store_true', help = 'mask a percentage of the negative examples to rebalance data')

    parser.add_argument('--mask_probability', '-msk_pr', default = 0.9, type = float, help = 'if used, the percentage of negative examples to mask (see above)')

    parser.add_argument('--standard_split', '-split', type = str, help = 'If included, use a pre-defined train/development/test split instead of cross validation. The value should be that of a valid json file containing the splits in the form of a list of file names associated to train, validation and test keys respectively')

    parser.add_argument('--self_attention_window', '-window', default = 120, type = int, help = 'the value of window for restricted transformers (i.e. LongT5 and Longformer).')

    parser.add_argument('--both_datasets', '-bd', action = 'store_true', help = 'if included, use both NonNews and RadioNews datasets')

    parser.add_argument('--infer', action = 'store_true', help = 'use the pre-trained model to test only: the train script must have been previously run, generating the pre-defined folder structure.')

    parser.add_argument('--timing_file', required=False, type=str, help="If included, the pickled file containing the timings information per sentence of the given dataset.")

    args = parser.parse_args()

    main(args)    