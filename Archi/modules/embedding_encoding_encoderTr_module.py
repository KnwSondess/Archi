import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from Archi.modules.encodings import PosEncoding, PosLearnedEncoding, TokenLearnedEncoding
from Archi.modules.embedding import Embedding
from typing import Dict, List
import torch
from torch import nn, einsum
from Archi.modules.module import Module

class EmbdEncMultiTransf(nn.Module):
    def __init__(self, args):
        '''
        Embedding, ecoder for language, frames  or action inputs, and multi-layer encoder transformer
        '''

         super(EmbdEncMultiTran, self).__init__(id=id,
                                                type="EmbeddingEncodingTransformerEncoder",
                                                config=config,
                                                input_stream_ids=input_stream_ids,)

        
        #Embedding
        self.embedding = Embedding(args.embd) if args.embd['lg'] else None 
        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb, args.encoder_heads, args.demb,
            args.dropout['transformer']['encoder'])
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer, args.encoder_layers)

        # how many last actions to attend to
        self.num_input_actions = args.num_input_actions

        # encodings
        self.enc_pos = PosEncoding(args.demb) if args.enc['pos'] else None
        self.enc_pos_learn = PosLearnedEncoding(args.demb) if args.enc['pos_learn'] else None
        self.enc_token = TokenLearnedEncoding(args.demb) if args.enc['token'] else None
        self.enc_layernorm = nn.LayerNorm(args.demb)
        self.enc_dropout = nn.Dropout(args.dropout['emb'], inplace=True)

        #def forward(self,)
        #No forward for this module
      
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        for key, experiences in input_streams_dict.items():
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]
            batch_size = experiences.size(0)

            experiences = experiences.view(batch_size, -1)
            if self.use_cuda:   experiences = experiences.cuda()

        model = {
            'embedding':self_embedding, 
            'preprocessor':self.encoding,
            'processor':self.enc_transformer,
        }
            model = self.layers(experiences)
            outputs_stream_dict[f'emb_{key}'] = model['embedding']
            outputs_stream_dict[f'processed_{key}'] = model['preprocessor']
            outputs_stream_dict[f'proc_{key}'] = model['processor']
            outputs_stream_dict[f'processed_{key}'] = features

        return outputs_stream_dict 
