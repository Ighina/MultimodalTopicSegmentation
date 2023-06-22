# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:00:49 2021

@author: Iacopo
"""

import torch
import pytorch_lightning as pl
import numpy as np
from scipy.stats import mode
from models.CRF import *
import segeval
from sklearn.metrics import f1_score

def get_boundaries(boundaries):
    tot_sents = 0
    masses = []
    for boundary in boundaries:
        tot_sents += 1
        if boundary:
            masses.append(tot_sents)
            tot_sents = 0
    return masses

def compute_Pk(boundaries, ground_truth, window_size = None, boundary_symb = '1'):
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    
    if window_size is None:
        result = segeval.pk(h, t)
    else:
        result = segeval.pk(h,t, window_size = window_size)
    
    boundaries[-1] = 0
    ground_truth[-1] = 0
    return result

def compute_window_diff(boundaries, ground_truth, window_size = None, 
                            segval = True, boundary_symb = '1'):
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    
    if window_size is None:
        result = segeval.window_diff(h, t)
    else:
        result = segeval.window_diff(h,t, window_size = window_size)
    
    boundaries[-1] = 0
    ground_truth[-1] = 0
    return result

def WinPR(reference, hypothesis, k = 10):
    """
    Implementation of the metric by scaiano et al. 2012 (https://aclanthology.org/N12-1038.pdf)
    
    Parameters
    ----------
    reference : list of int
        the reference segmentation (e.g. [0,0,0,1,0,0).
    hypothesis : list of int
        the hypothesised segmentation (e.g. [0,0,1,0,0,0]).
    k : int, optional
        The window value as defined in scaiano et al. 2012. The default is 10.

    Returns
    -------
    Precision, Recall and F1 measures (floats).

    """
    assert len(reference)==len(hypothesis), "Hypothesis and reference should be the same length!"
    
    N = len(reference)
    
    RC = []
    Spans_R = []
    Spans_C = []
    
    for i in range(1-k, N+1):
        prev_br = 0
        prev_bc = 0
        
        try:
            if Spans_R[-1][0] == 1:
                prev_br = 1
        except IndexError:
            pass
        try:
            if Spans_C[-1][0] == 1:
                prev_bc = 1
        except IndexError:
            pass
        
        Spans_R.append(reference[i:i+k])
        Spans_C.append(hypothesis[i:i+k])
        
        R = sum(reference[max(i,0):i+k])+prev_br
        C = sum(hypothesis[max(i,0):i+k]) + prev_bc
        
        RC.append((R,C))
    
    # RC = [(sum(reference[i:i+k]), sum(hypothesis[i:i+k])) for i in range(1-k, N+1)]
    
    TP =  sum([min(R, C) for R, C in RC])
    
    TN = -k*(k-1) + sum([k - max(R, C) for R, C in RC])
    
    FP = sum([max(0, C - R) for R, C in RC])
    
    FN = sum([max(0, R - C) for R, C in RC])
    try:
        precision = TP/(TP+FP)
    except ZeroDivisionError:
        return 0, 0, 0
        
    recall = TP/(TP+FN)
    
    f1 = 2*(precision*recall/(precision+recall))
    
    return precision, recall, f1

def B_measure(boundaries, ground_truth):
    """
    Boundary edit distance-based methods for text segmentation evaluation (Fournier2013)
    """
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    # value errors occur when there is no boundary in the reference segmentation
    # if len(t)<2:     
    cm = segeval.boundary_confusion_matrix(h, t, n_t = 4)
    b_precision = float(segeval.precision(cm, classification = 1))
    b_recall = float(segeval.recall(cm, classification = 1))
    try:
        b_f1 = 2*(b_precision*b_recall)/(b_precision+b_recall)
    except ZeroDivisionError:
        b_f1 = 0.0
    # b_f1 = segeval.fmeasure(cm, classification = 1)
    # else:
    #    b_precision = 0
    #    b_recall = 0
    #    b_f1 = 0
    # if len(t)<2:
    b = segeval.boundary_similarity(h, t, n_t = 10)
    # else:
    #    b = 0
    return float(b_precision), float(b_recall), float(b_f1), float(b)
    

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


class TextSegmenter(pl.LightningModule):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers = 1,
                 batch_first = True, LSTM = True, bidirectional = True, architecture = 'biLSTMCRF',
                 lr = 0.01, dropout_in = 0.0, dropout_out = 0.0, optimizer = 'SGD', 
                 positional_encoding = True, nheads = 8, end_boundary = False, threshold = None,
                 search_threshold = False, metric = 'Pk', cosine_loss = False, zero_baseline = False, loss_fn = 'CrossEntropy',
                 no_validation = False, all_results = False, all_scores = False, alpha=0.9, gamma=2, attention_window = 120, switch = 'dense'):
        super().__init__()
        self.validation = not no_validation
        self.cos = cosine_loss

        self.double_input = False # fusion architectures (except early fusion)
        self.domain = False # domain adaptation
        self.delete_last_target = False
        
        if architecture == 'biLSTMCRF':
          self.cos = False
          self.model = BiRnnCrf(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 architecture = 'rnn')
        elif architecture == 'SimpleBiLSTM':
          self.model = SimpleBiLSTM(embedding_dim, hidden_dim, num_layers)
        elif architecture == 'MLP':
          self.model = MLP(embedding_dim, hidden_dim, num_layers)
        elif architecture == 'Transformer-CRF':
          self.cos = False
          self.model = TransformerCRF(tagset_size, embedding_dim, hidden_dim, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, num_layers = num_layers, positional_encoding = positional_encoding, nheads = nheads)
        elif architecture == 'BiLSTM':
          self.model = BiLSTM(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 loss_fn = loss_fn, threshold = threshold, alpha=alpha, gamma=gamma)
        elif architecture == 'Transformer':
          self.model = Transformer_segmenter(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = loss_fn, positional_encoding = positional_encoding, nheads = nheads, threshold = threshold, alpha=alpha, gamma=gamma, window_size = attention_window)
        elif architecture == 'RecurrentLongT5':
          self.model = RecurrentLongT5(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = loss_fn, nheads = nheads, threshold = threshold, alpha=alpha, gamma=gamma, window_size = attention_window)
        elif architecture == 'BiLSTMRestrictedMHA':
            self.model = RecurrentLongformer(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = loss_fn, nheads = nheads, threshold = threshold, alpha=alpha, gamma=gamma, window_size = attention_window)
        elif architecture == 'BiLSTMLateFusion':
            self.model = BiLSTMLateFusion(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 loss_fn = loss_fn, threshold = threshold, alpha=alpha, gamma=gamma)
            
            self.double_input = True
        elif architecture == 'SwitchBiLSTM':
            
            bias_adapt = False
            switch_lstm_adapt = False
            switch_dense_adapt = False

            if switch == "lstm":
                switch_lstm_adapt = True
            elif switch == 'dense':
                switch_dense_adapt = True
            elif switch == 'bias':
                raise NotImplementedError()

            self.model = SwitchBiLSTM(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 loss_fn = loss_fn, threshold = threshold, alpha=alpha, gamma=gamma, 
                 bias_adapt = bias_adapt, switch_lstm_adapt = switch_lstm_adapt, switch_dense_adapt = switch_dense_adapt)
            
            self.domain = True
        
        elif architecture == 'SheikhBiLSTM':
            self.model = SheikhBiLSTM(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 dropout_in = 0.5, dropout_attention = 0, batch_first = True)
        
        else:
          raise ValueError("No other architectures implemented yet")
        self.learning_rate = lr
        self.optimizer = optimizer
        self.eb = end_boundary
        self.threshold = threshold
        self.s_th = search_threshold
        self.metric = metric
        self.best_th = []
        self.losses = []
        self.targets = []
        self.zero_base = zero_baseline
        self.all = False
        if all_results:
            self.all = True
            self.results = []
        self.all_scores = False
        if all_scores:
            self.all_scores = True
            self.scores = [] 
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        if self.cos:
            segments = batch['src_segments']
        else:
            segments = None
        
        if self.double_input:
            sentence2 = batch['src_tokens2']

        if self.domain:
            domain = batch['domain']
        
        self.best_th = []
        self.losses = []
        self.targets = []
        if self.domain:
            try:
                loss = self.model.loss(sentence, lengths, target, domain, segments = segments)
            except TypeError:
                loss = self.model.loss(sentence, lengths, target, domain)
        elif self.double_input:
            try:
                loss = self.model.loss(sentence, sentence2, lengths, target, segments = segments)
            except TypeError:
                loss = self.model.loss(sentence, sentence2, lengths, target)
        else:
            try:
                loss = self.model.loss(sentence, lengths, target, segments = segments)
            except TypeError:
                loss = self.model.loss(sentence, lengths, target)
        
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     sentence = batch['src_tokens'] 
    #     target = batch['tgt_tokens']
    #     lengths = batch['src_lengths']
        
    #     loss = self.model.loss(sentence, lengths, target)
        
    #     self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']

        if self.double_input:
            sentence2 = batch['src_tokens2']

        if self.domain:
            domain = batch['domain']
        
        if self.s_th:
            if self.domain:
                scores, tags = self.model(sentence, lengths, domain)
            elif self.double_input:
                scores, tags = self.model(sentence, sentence2, lengths)
            else:
                scores, tags = self.model(sentence, lengths)

            for index, score in enumerate(scores):
                self.losses.append(score[:lengths[index]].detach().cpu().numpy())
                self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
            
        else:
            if self.domain:
                loss = self.model.loss(sentence, lengths, target, domain)
            elif self.double_input:
                loss = self.model.loss(sentence, sentence2, lengths, target)
            else:
                loss = self.model.loss(sentence, lengths, target)
            self.log_dict({'val_loss': loss, 'threshold': 0.5})
            return loss
            # threshold = 0.5
            # score, tags = self.model(sentence, lengths)
            
            # if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
            #     loss_precision = 0
            #     loss_recall = 0
            #     loss_f1 = 0
            #     loss_b = 0
            # else:
            #     loss_PK = 0
            #     loss_F1 = 0
            #     loss_WD = 0
            
            
            # for i, tag in enumerate(tags):
                
            #     if self.eb:
            #         tag[-1]=0
            #         target[i][-1]=0
                
            #     if self.metric.lower()=='b':
            #         precision, recall, f1 , b = B_measure(tag, target[i][:lengths[i]].detach().cpu().numpy())

            #         loss_precision += precision
            #         loss_recall += recall
            #         loss_f1 += f1
            #         loss_b += b

            #     elif self.metric.lower()=='scaiano':
            #         precision, recall, f1 = WinPR(tag, target[i][:lengths[i]].detach().cpu().numpy())

            #         loss_precision += precision
            #         loss_recall += recall
            #         loss_f1 += f1
                    
            #     else:
            #         loss_PK += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                        
            #         loss_F1 += f1_score(target[i][:lengths[i]].detach().cpu().numpy().astype(int), np.array(tag).astype(int),
            #                                 labels = [1], average = None)
                  
            #         try:
            #             loss_WD += float(compute_window_diff(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
            #         except AssertionError:
            #             loss_WD += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
            
            # if self.metric.lower() == 'b' or self.metric.lower()=='scaiano':
            #     results = {'b_precision': loss_precision/len(target), 'b_recall': loss_recall/len(target), 'b_f1': loss_f1/len(target), 'threshold': threshold}
            #     if self.metric.lower()=='b':
            #         val_loss = loss_b/len(target)
            #         results['val_loss'] = val_loss
            #     else:
            #         val_loss = results[-1].pop('b_f1')
            #         results['val_loss'] = val_loss
            # else:
            #     try:
            #         results = {'Pk_loss': loss_PK/len(target), 
            #            'F1_loss': (loss_F1/len(target))[0],
            #            'WD_loss': loss_WD/len(target),
            #            'threshold': threshold}
            #     except:
            #         results = {'Pk_loss': loss_PK/len(target), 
            #            'F1_loss': loss_F1/len(target),
            #            'WD_loss': loss_WD/len(target),
            #            'threshold': threshold}
                
            #     if self.metric=='F1':
            #         val_loss = results.pop('F1_loss')
            #         results['val_loss'] = val_loss
                    
            #     elif self.metric == 'WD':
            #         val_loss = results.pop('WD_loss')
            #         results['val_loss'] = val_loss
            
            #     else:
            #         val_loss = results.pop('Pk_loss')
            #         results['val_loss'] = val_loss
            
            # self.log_dict(results, on_epoch = True, prog_bar=True)
            # return val_loss
            
        
    
    # def on_validation_epoch_end(self):
    def uncomment_above_to_use_koshorek(self):
        if self.s_th:
            scores = self.losses
            target = self.targets
            thresholds = np.arange(0.05,1,0.05)
            
            results = []
            best_idx = 0
            if self.metric.lower()=='pk' or self.metric.lower()=='wd':
                best = 1
            else:
                best = -1
            for index, th in enumerate(thresholds):
                if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
                    loss_precision = 0
                    loss_recall = 0
                    loss_f1 = 0
                    loss_b = 0
                else:
                    loss_PK = 0
                    loss_F1 = 0
                    loss_WD = 0
                
                
                for i, tag in enumerate(scores):
                    tag = tag[:,1]>th
                    
                    if self.eb:
                        tag[-1]=0
                        target[i][-1]=0
                    
                    if self.metric.lower()=='b':
                        precision, recall, f1 , b = B_measure(tag, target[i])

                        loss_precision += precision
                        loss_recall += recall
                        loss_f1 += f1
                        loss_b += b

                    elif self.metric.lower()=='scaiano':
                        precision, recall, f1 = WinPR(tag, target[i])

                        loss_precision += precision
                        loss_recall += recall
                        loss_f1 += f1
                    
                    else:
                        loss_PK += float(compute_Pk(np.array(tag), target[i]))
                        
                        loss_F1 += f1_score(target[i].astype(int), np.array(tag).astype(int),
                                            labels = [1], average = None)
                  
                        try:
                            loss_WD += float(compute_window_diff(np.array(tag), target[i]))
                        except AssertionError:
                            loss_WD += float(compute_Pk(np.array(tag), target[i]))
                
                if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
                    results.append({'b_precision': loss_precision/len(target), 'b_recall': loss_recall/len(target), 'b_f1': loss_f1/len(target)})
                    if self.metric.lower()=='b':
                        val_loss = loss_b/len(target)
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                    else:
                        val_loss = results[-1].pop('b_f1')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th

                else:
                    try:
                        results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': (loss_F1/len(target))[0],
                           'WD_loss': loss_WD/len(target)})
                    except:
                        results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': loss_F1/len(target),
                           'WD_loss': loss_WD/len(target)})
                    
                    if self.metric=='F1':
                        val_loss = results[-1].pop('F1_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                        
                    elif self.metric == 'WD':
                        val_loss = results[-1].pop('WD_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss<best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                
                    else:
                        val_loss = results[-1].pop('Pk_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss<best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
            try:
                # if results[best_idx]['valid_loss'] == 0 or results[best_idx]['valid_loss'] == 1 and self.best_th==0.05:
                #    self.best_th = 0.5
                results[best_idx]['threshold'] = self.best_th
                if results[best_idx]['threshold'] is None:
                  results[best_idx]['threshold'] = 0.4
                
            except IndexError:
                results[best_idx]['threshold'] = 0.4
            
            self.log_dict(results[best_idx], on_epoch = True, prog_bar=True)
            
        
        

    def test_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']

        if self.double_input:
            sentence2 = batch['src_tokens2']

        if self.domain:
            domain = batch['domain']
        
        if self.s_th:
            raise NotImplementedError()
            # scores, tags = self.model(sentence, lengths)
            # for index, score in enumerate(scores):
            #     self.losses.append(score[:lengths[index]].detach().cpu().numpy())
            #     self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
        else:
            if self.zero_base:
                threshold = 0.4
                pad_tags = np.zeros((sentence.shape[0], sentence.shape[1]))
                tags = []
                for index, tag in enumerate(pad_tags):
                    tags.append(tag[:lengths[index]])
            else:
                threshold = self.threshold if self.threshold is not None else .4
                if not threshold:
                    threshold = 0.5
                
                self.model.th = threshold

                if self.domain:
                    score, tags = self.model(sentence, lengths, domain)
                elif self.double_input:
                    score, tags = self.model(sentence, sentence2, lengths)
                else:
                    score, tags = self.model(sentence, lengths)
            
            if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
                loss_precision = 0
                loss_recall = 0
                loss_f1 = 0
                loss_b = 0
            else:
                loss_PK = 0
                loss_F1 = 0
                loss_WD = 0
            
            
            for i, tag in enumerate(tags):
                
                if self.eb:
                    tag[-1]=0
                    target[i][-1]=0
                
                if self.metric.lower()=='b':
                    precision, recall, f1 , b = B_measure(tag, target[i][:lengths[i]].detach().cpu().numpy())

                    loss_precision += precision
                    loss_recall += recall
                    loss_f1 += f1
                    loss_b += b

                elif self.metric.lower()=='scaiano':
                    precision, recall, f1 = WinPR(tag, target[i][:lengths[i]].detach().cpu().numpy())

                    loss_precision += precision
                    loss_recall += recall
                    loss_f1 += f1
                    
                else:
                    loss_PK += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                        
                    loss_F1 += f1_score(target[i][:lengths[i]].detach().cpu().numpy().astype(int), np.array(tag).astype(int),
                                            labels = [1], average = None)
                  
                    try:
                        loss_WD += float(compute_window_diff(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                    except AssertionError:
                        loss_WD += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
            
            if self.metric.lower() == 'b' or self.metric.lower()=='scaiano':
                results = {'b_precision': loss_precision/len(target), 'b_recall': loss_recall/len(target), 'b_f1': loss_f1/len(target), 'threshold': threshold}
                if self.metric.lower()=='b':
                    val_loss = loss_b/len(target)
                    results['test_loss'] = val_loss
                else:
                    val_loss = results[-1].pop('b_f1')
                    results['test_loss'] = val_loss
            else:
                try:
                    results = {'Pk_loss': loss_PK/len(target), 
                       'F1_loss': (loss_F1/len(target))[0],
                       'WD_loss': loss_WD/len(target),
                       'threshold': threshold}
                except:
                    results = {'Pk_loss': loss_PK/len(target), 
                       'F1_loss': loss_F1/len(target),
                       'WD_loss': loss_WD/len(target),
                       'threshold': threshold}
                
                if self.metric=='F1':
                    val_loss = results.pop('F1_loss')
                    results['test_loss'] = val_loss
                    
                elif self.metric == 'WD':
                    val_loss = results.pop('WD_loss')
                    results['test_loss'] = val_loss
            
                else:
                    val_loss = results.pop('Pk_loss')
                    results['test_loss'] = val_loss
            
            if self.all:
                self.results.append(results)
            if self.all_scores:
                self.scores.extend([s.detach().cpu().numpy() for s in score])
            
            self.log_dict(results, on_epoch = True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        sentence = batch['src_tokens']
        lengths = batch['src_lengths']
        
        score, tags = self.model(sentence, lengths)
        return tags
        
    
    # def on_test_epoch_end(self):
    #     if self.s_th:
    #         scores = self.losses
    #         target = self.targets
    #         thresholds = [.05, .1, .2, .3, .4, .5, .6]
            
    #         results = []
    #         best_idx = 0
    #         best = 0 if self.metric == 'F1' else 1
    #         for index, th in enumerate(thresholds):
                
    #             loss_PK = 0
    #             loss_F1 = 0
    #             loss_WD = 0
                
                
    #             for i, tag in enumerate(scores):
    #                 tag = tag[:,1]>th
                    
    #                 loss_PK += float(compute_Pk(np.array(tag), target[i]))
    #                 loss_F1 += f1_score(target[i].astype(int), np.array(tag).astype(int),
    #                                         labels = [1], average = None)
                  
    #                 loss_WD += float(compute_window_diff(np.array(tag), target[i]))
                
                
    #             try:
    #                 results.append({'Pk_loss': loss_PK/len(target), 
    #                       'F1_loss': (loss_F1/len(target))[0],
    #                       'WD_loss': loss_WD/len(target)})
    #             except:
    #                 results.append({'Pk_loss': loss_PK/len(target), 
    #                       'F1_loss': loss_F1/len(target),
    #                       'WD_loss': loss_WD/len(target)})
                    
    #             if self.metric=='F1':
    #                 val_loss = results[-1].pop('F1_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss>best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
                        
    #             elif self.metric == 'WD':
    #                 val_loss = results[-1].pop('WD_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss<best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
                
    #             else:
    #                 val_loss = results[-1].pop('Pk_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss<best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
    #         try:
    #             if results[best_idx]['test_loss'] == 0 or results[best_idx]['test_loss'] == 1 and self.best_th==0.05:
    #                 self.best_th = 0.5
    #             results[best_idx]['threshold'] = self.best_th
    #             if results[best_idx]['threshold'] is None:
    #               results[best_idx]['threshold'] = 0.4
                
    #         except IndexError:
    #             results[best_idx]['threshold'] = 0.4
            
    #         self.log_dict(results[best_idx])
        
    #     else:
    #         self.log_dict()
            
    def configure_optimizers(self):
        if self.optimizer == 'SGD':
          optimizer = torch.optim.SGD(self.parameters(),
                                     lr=self.learning_rate, weight_decay = 1e-4, momentum = 0.9)
        else:
          optimizer = torch.optim.Adam(self.parameters(), eps = 1e-7,
                                     lr=self.learning_rate)

        # return {'optimizer': optimizer}
        
        if self.metric.lower()=='pk' or self.metric.lower()=='wd' or not self.s_th:
            if self.validation:        
                scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = .8, patience = 10), "monitor": "val_loss"}
            else:
                scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = .8, patience = 10), "monitor": "training_loss"}
        else:
            if self.validation:
                scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = .8, patience = 10), "monitor": "val_loss"}
            else:
                scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = .8, patience = 10), "monitor": "training_loss"}   
         
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
