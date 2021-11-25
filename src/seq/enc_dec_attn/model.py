import math
from time import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from seq import USE_CUDA


# *** globals ***


# *** functions ***
def make_model(src_vocab: int, tgt_vocab: int, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab)
    )

    return model.cude() if USE_CUDA else model


def run_epoch(data_iter, model: 'EncoderDecoder', loss_compute, print_every=50):
    start = time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)

        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time() - start
            print(f"Epoch step: {i} Loss: {loss / batch.nseqs, print_tokens / elapsed}")
            start = time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


# *** classes ***
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)
                              )
        loss = loss / norm

        if self.opt is not None:
            loss.backward()


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
        Layer to consume matrix of hidden states and output probabilities over the vocabulary TODO: Check

    """

    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', src_embed: nn.Embedding, trg_embed: nn.Embedding,
                 generator: 'Generator'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Encode and decode the masked source and target sequences

        Parameters
        ----------
        src:
            Batched input sequence embeddings, shape [batch size, sequence length, input_size]
        trg:
            TODO
        src_mask:
            TODO
        trg_mask:
            TODO
        src_lengths:
            Sequence lengths for each input source sequence

        """
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask, decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final, src_mask, trg_mask,
                            hidden=decoder_hidden)


class Generator(nn.Module):
    """Define standard linear + softmax generation step to create probabilities"""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """
    Encodes a sequence of word embeddings

    Parameters
    ----------
    input_size: int
        The number of expected features in the input
    hidden_size: int
        The number of features in the hidden state h
    num_layers: int
        Number of recurrent layers. Stacks RNNs together.
    dropout: float
        If >0 introduces a Dropout layer on the outputs of each GRU layer except the last layer, with the dropout
        probability being `dropout`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.):
        super().__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """Applies a bidirectional GRU to a sequence of embeddings

        Parameters
        ----------
        x:
            Padded input batch of variable length sequences. Must be sorted by length in decreasing order.
            Dimensions [batch, time, dim].
        mask:
            TODO: Doesn't get used!
        lengths: torch.Tensor or list
            List of sequence lengths of each batch element. It is assumed that sequences are padded after
            iterating through the corresponding number of elements of the sequence.

        """
        # Packed padded sequence allows us to pass mini-batches to an RNN, and have the RNN stop updating the hidden
        # state once we have reached the end of the sequence. Has greater GPU efficiency.
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # inverse of pack_padded_sequence

        # Concatenate the final states for both RNN directions. This is a summary of the entire sentence and will
        # be used as input to the decoder.
        # TODO: Check, not sure about this.
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # TODO: check dim

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention TODO: conditional? Generally need to understand this

    Parameters
    ----------
    emb_size:
        TODO
    hidden_size:
        TODO
    attention:
        TODO
    num_layers:
        TODO
    dropout:
        TODO
    bridge:
        TODO
    """

    def __init__(self, emb_size: int, hidden_size: int, attention: 'BahdanauAttention', num_layers: int = 1,
                 dropout: float = 0.5,
                 bridge: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embedded, encoder_hidden, src_mask, proj_key, hidden: torch.Tensor):
        """Perform a single decoder step (1 word). This is used for inference.

        Parameters
        ----------
        prev_embedded:
            TODO
        encoder_hidden:
            TODO
        src_mask:
            TODO
        proj_key:
            TODO
        hidden:
            TODO

        Returns
        -------
        output:
            TODO
        hidden:
            TODO
        pre_output:
            TODO

        """
        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [num layers, B, D] -> [B, 1, D]
        # TODO: explain
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask
        )

        # update rnn hidden state
        rnn_input = torch.cat([prev_embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embedded, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final, src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time. This is used for training.

        Parameters
        ----------
        trg_embed:
            Target embedding. We use this for teacher forcing
        encoder_hidden:
            TODO
        encoder_final:
            TODO
        src_mask:
            TODO
        trg_mask:
            TODO
        hidden:
            TODO
        max_len:
            TODO

        """

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            # During training, by using the target embedding rather than the prediction, we can get faster learning:
            # this is called teacher forcing.
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Return initial decoder state, conditioned on the final encoder state"""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention:
    """Implements Bahdanau (MLP) attention

    Parameters
    ----------
    hidden_size:
        TODO
    key_size:
        TODO
    query_size:
        TODO
    """

    def __init__(self, hidden_size, key_size: Optional[int] = None, query_size=None):
        super().__init__()

        key_size = 2 * hidden_size if key_size is None else key_size  # assume a bi-directional encoder
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.alphas = None  # for storing attention scores

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        """
        Parameters
        ----------

        """
        assert mask is not None, "mask is required"  # TODO: ganky

        # Project the query (decoder state)
        query = self.query_layer(query)

        # Calculate scores
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        #  Mask invalid positions
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas  # [B, 1, M]

        context = torch.bmm(alphas, value)  # [B, 1, 2D]

        return context, alphas
