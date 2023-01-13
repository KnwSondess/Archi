
'''
code adapted from ET paper
https://github.com/alexpashevich/E.T./tree/92ee2378d596b55f05e5c1949726577a64215f04
'''

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderLayerModel(nn.Module):  

   def __init__(self, args):
        '''
        transformer encoder for language, frames and action inputs
        '''
        super(TransformerEncoderLayerModel, self).__init__()

        # 
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb, args.encoder_heads, args.demb,
            args.dropout['transformer']['encoder'])
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer, args.encoder_layers)

