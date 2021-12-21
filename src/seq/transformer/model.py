"""Based on (but not identical to) "The Annotated Transformer"

see http://nlp.seas.harvard.edu/2018/04/03/attention.html
and https://arxiv.org/pdf/1706.03762.pdf
"""

import copy
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# *** globals ***
Sizes = namedtuple('Sizes', ['batch', 'n_batches', 'src_vocab', 'tgt_vocab', 'n_layers', 'd_model', 'd_ff', 'h'])


# *** functions ***
def clones(module: nn.Module, N: int):
    """Produce N identical copies of a layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'. This is a faster/more memory efficient alternative to additive attention
    in Bahdanau. TODO: Understand this better. This omits the alignment model.
    """
    d_k = query.size(-1)
    # scaling controls the variance of the dot-product, so we control the vanishing gradient problem
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def make_model(sizes: 'Sizes', dropout=0.1):

    c = copy.deepcopy
    attn = MultiHeadedAttention(sizes.h, sizes.d_model)
    ff = PositionwiseFeedForward(sizes.d_model, sizes.d_ff, dropout)


    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                sizes.d_model, c(attn), c(ff), dropout),
            sizes.n_layers,
            sizes),
        Decoder(
            DecoderLayer(
                sizes.d_model, c(attn), c(attn), c(ff), dropout),
            sizes.n_layers,
            sizes),
        Generator(sizes.d_model, sizes.tgt_vocab, sizes),
        sizes,
        dropout=dropout,
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model



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
        TODO
    generator: Generator
        Layer to consume matrix of decoder pre_output hidden states, and output probabilities over the vocabulary

    """

    def __init__(self, encoder: 'Encoder', decoder: 'Decoder',
                 generator: 'Generator', sizes: 'Sizes', dropout: float = 0.0):
        super().__init__()

        c = copy.deepcopy

        self.encoder = encoder
        self.decoder = decoder

        position = PositionalEncoding(sizes.d_model, dropout)
        self.src_embed = nn.Sequential(Embeddings(sizes.d_model, sizes.src_vocab), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(sizes.d_model, sizes.tgt_vocab), c(position))

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
        # assert src.shape == (self.sizes.batch, self.sizes.src_vocab - 1)  # todo
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

    def __init__(self, layer: nn.Module, N: int, sizes: Sizes):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.sizes = sizes

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Pass the input (and mask) through each layer in turn

        Parameters
        ----------
        x: torch.Tensor
            The batched input tokens
        mask: torch.Tensor
            Non-padded sequences

        Returns
        -------
        torch.Tensor:
            Encoded inputs
        """
        # assert x.shape == (self.sizes.batch, self.sizes.src_vocab)  # todo

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


class Decoder(nn.Module):
    """Generic N layer decoder with masking"""

    def __init__(self, layer, N, sizes: 'Sizes'):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.sizes = sizes

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections"""
        m = memory
        x = self.sublayer[0](x, lambda i: self.self_attn(i, i, i, tgt_mask))
        x = self.sublayer[1](x, lambda i: self.src_attn(i, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # Assume d_v == d_k  # TODO: ?
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2. TODO: Understand better"""
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))
                             ]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear layer
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement positional encoding.

    Adds a sinusoid based on position. The frequency and offset of the wave is different for each dimension.

    See https://arxiv.org/pdf/1705.03122.pdf for more details
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)
