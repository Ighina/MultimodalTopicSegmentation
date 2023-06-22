import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence as PAD
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from collections import OrderedDict

def create_mask(src, lengths):
    """Create a mask hiding future tokens
    Parameters:
        src (tensor): the source tensor having shape [batch_size, number_of_steps, features_dimensions]
        length (list): a list of integers representing the length (i.e. number_of_steps) of each sample in the batch."""
    mask = []
    max_len = src.shape[1]
    for index, i in enumerate(src):
        # The mask consists in tensors having false at the step number that doesn't need to be hidden and true otherwise
        mask.append([False if (i+1)>lengths[index] else True for i in range(max_len)])
    return torch.tensor(mask)

class RNN(nn.Module):
    """Class implementing recurrent networks (LSTM/GRU)"""
    def __init__(self, embed_size,hidden_size,num_layers=1,labels=1,
                  bidirectional=False, dropout_in=0.0,
                  dropout_out=0.0,padding_idx=0, batch_first=True,
                  LSTM=True):
        super(RNN,self).__init__()
        self.embed_size = embed_size # embedding/input dimensions
        self.hidden_size = hidden_size # hidden layers' dimensions
        self.labels = labels # output classes
        self.num_layers = num_layers # number of recurrent layers
        self.bidirectional=bidirectional # boolean: bidirectional=True makes the network bidirectional
        
        
        if LSTM:
            # If LSTM is true, use LSTM else use GRU
            self.rnn = nn.LSTM(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
            
        else:
            self.rnn = nn.GRU(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
        
        self.dropout_in = dropout_in
        
        self.dropout_out = dropout_out
        
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization: https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
        
        
        
    def forward(self, line, line_len=None, apply_softmax=False, return_final=False, classifier=False):
        """
        Parameters:
            line (tensor): the input tensor having shape [batch_size, number_of_steps, features_dimensions]
            line_len (list): a list containing the length of each sample in the batch. If no list is passed, then the function assumes all samples to have same length (i.e. no padding)
            apply_softmax (boolean): whether to apply the softmax function or not (as in the case for cross-entropy loss) after the classifier layer
            return_final (boolean): whether or not to return the final hidden state (e.g. to use it as first hidden state in a decoder)
            classifier (boolean): whether the network has a classifier layer or it acts just as an encoder"""
        
        if self.dropout_in:
            # if dropout_in value is not 0, then apply the dropout
            line = F.dropout(line, p=self.dropout_in)
        
        if line_len is not None:
            # if lengths are provided, pack the input tensor, else nothing happens
            embedded = PACK(line, line_len.data.tolist(), batch_first=True, enforce_sorted=False)
        else:
            embedded = line
        
        if self.bidirectional:
            # if the network is bidirectional, first create the initial hidden and memory cell states (for LSTM)
            batch_size = line.shape[0]
            
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
            
            
            hidden_initial = line.new_zeros(*state_size)
            
            cells_initial = line.new_zeros(*state_size)
            
            packed_out, (final_hidden_states, final_cell_states) = self.rnn(embedded,(hidden_initial,cells_initial))
            
            rnn_out, _ = PAD(packed_out, batch_first=True) # unpack the rnn output and pad it with 0s where needed
            
            if self.dropout_out:
                # if dropout_out is not 0, apply dropout to the rnn output
                rnn_out = F.dropout(rnn_out, p=self.dropout_out)            
            
            
            if return_final:
                # if returning final hidden state, concatenate the final hidden state from forward and backward layers of the network
                def combine_directions(outs):
                    return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
                final_hidden_states = combine_directions(final_hidden_states)
                final_cell_states = combine_directions(final_cell_states)
                return rnn_out, (final_hidden_states, final_cell_states)
            
            else:
                return rnn_out
            
        
        else:
            # same as the bidirectional case, but with less operations needed
            
            rnn_out,h_n = self.rnn(embedded)
            
            if self.dropout_out:
                rnn_out = F.dropout(rnn_out, p=self.dropout_out)
                                    
            if return_final:
                return rnn_out, h_n
            else:
                return rnn_out
    
def column_gatherer(y_out, lengths):
    """Gather the final states from a RNN with padded inputs, so that the
    actual final state for each sample can be used for classification.
    Parameters:
        y_out (tensor): the RNN output
        lengths (tensor): the individual lengths of each sample in the batch"""
    lengths = lengths.long().detach().numpy()-1
    out = []
    for batch_index, column_index in enumerate(lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=500):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         for pos in range(max_len):
#               for i in range(0, d_model, 2):
#                   pe[pos, i] = \
#                   math.sin(pos / (10000 ** ((2 * i)/d_model)))
#                   try:
#                     pe[pos, i + 1] = \
#                     math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
#                   except IndexError:
#                     pass
#         self.register_buffer('pe', pe, persistent=False)
    
#     def forward(self, x):
#         x = x + self.pe[:x.size(1), :]
#         return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    """Class implementing transformer ecnoder, partially based on
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, in_dim, h_dim, n_heads, n_layers, dropout=0.2, drop_out = 0.0):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(in_dim, n_heads, h_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers, norm=nn.LayerNorm(in_dim))
        # norm=nn.LayerNorm(in_dim)
        self.in_dim = in_dim
        self.drop_out = drop_out
        
    def forward(self, src, line_len=None):
        src = src * math.sqrt(self.in_dim)
        # src = self.pos_encoder(src)
        if line_len is not None:
            mask = create_mask(src, line_len).to(src.device)
        else:
            mask = None
        N, S = src.shape[0], src.shape[1]
        src = src.contiguous().view(S, N,-1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask = mask)
        if self.drop_out:
            output = F.dropout(output, p = self.drop_out)
        output = output.contiguous().view(N, S,-1)
        src = src.contiguous().view(N, S, -1)
        return src, output
    
class ConvolNet(nn.Module):
    """A simple single convolution layer for input preprocessing"""
    def __init__(self, in_dim, h_dim, kernel=1):
        super(ConvolNet, self).__init__()
        self.conv = nn.Conv1d(in_dim, h_dim, kernel)
        self.activation = nn.ReLU()
        
    def forward(self, src):
        return (src,self.activation(self.conv(src)))
        
class Convolutional(nn.Module):
    """A convolutional neural network for sequence tagging"""
    def __init__(self, in_dim, h_dim, n_layers, dropout=0.2, drop_out = 0.0, kernel=1):
        super(Convolutional, self).__init__()
        net = OrderedDict([('conv0', nn.Conv1d(in_dim, h_dim, kernel, padding = 'same')), ('activation0', nn.ReLU())])
        for i in range(1, n_layers):
            net['conv'+i] = nn.Conv1d(h_dim, h_dim, kernel, padding = 'same')
            net['activation'+i] = nn.ReLU()
        self.net = nn.Sequential(net)
        
    def forward(self, src):
        src = src.transpose(1,2)
        return self.net(src).transpose(1,2)