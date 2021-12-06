"""Based on (but not identical to) "The Annotated Transformer"

see http://nlp.seas.harvard.edu/2018/04/03/attention.html"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


# *** functions ***
def clones(module: nn.Module, N: int):
    """Produce N identical copies of a layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# *** classes ***

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.

    Parameters
    ----------
    encoder: Encoder
        Encodes a batched sequence of padded word embeddings, returning output for each input sequence and
        final hidden state (concatenation of forwards and backwards final RNN states)
    decoder: Decoder
    src_embed: nn.Module
        Embedding layer for the source
    trg_embed: nn.Module
        Embedding layer for the target
    generator: Generator
        Layer to consume matrix of decoder pre_output hidden states, and output probabilities over the vocabulary

    """

    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', src_embed: nn.Embedding, trg_embed: nn.Embedding,
                 generator: 'Generator', sizes: 'Sizes'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.sizes = sizes

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """Encode and decode the masked source and target sequences

        Parameters
        ----------
        src: torch.Tensor
            Batched input sequence embeddings, shape [batch size, sequence length]
        tgt: torch.Tensor
            Batched target sequence embeddings, shape [batch size, sequence length]
        src_mask: torch.Tensor
            Boolean array of elements that are not padding (src)
        tgt_mask: torch.Tensor
            Boolean array of elements that are not padding (trg)
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        """See Encoder.forward for details"""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """See Decoder.forward for details"""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step to create probabilities"""

    def __init__(self, d_model, vocab_size, sizes):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.sizes = sizes

    def forward(self, x):
        """

        Parameters
        ----------
        x:
            TODO

        Returns
        -------

        """
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module.

    Notes
    -----
    See https://arxiv.org/abs/1512.03385 for residual connection
    See https://arxiv.org/abs/1607.06450 for layer normalization
    TODO: Read and write up
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size.

        Parameters
        ----------
        x:
            TODO
        sublayer:
            TODO

        Returns
        -------

        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections"""
        x = self.sublayer[0](x, lambda i: self.self_attn(i, i, i, mask))
        return self.sublayer[1](x, self.feed_forward)
