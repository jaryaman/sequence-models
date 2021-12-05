"""
Based on https://bastings.github.io/annotated_encoder_decoder/
See https://arxiv.org/abs/1409.0473
"""

import math
from collections import namedtuple
from time import time
from typing import Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from seq import USE_CUDA
from seq.enc_dec_attn.parse import Batch

# *** globals ***

Sizes = namedtuple('Sizes',
                   ['src_vocab', 'tgt_vocab', 'emb', 'hidden', 'num_layers', 'batch', 'num_batches', 'sequence_length'])


# *** functions ***
def make_model(sizes: Sizes, dropout: float = 0.1):
    attention = BahdanauAttention(sizes)

    model = EncoderDecoder(
        Encoder(sizes.emb, sizes.hidden, sizes, num_layers=sizes.num_layers, dropout=dropout),
        Decoder(sizes.emb, sizes.hidden, attention, sizes, num_layers=sizes.num_layers, dropout=dropout),
        nn.Embedding(sizes.src_vocab, sizes.emb),
        nn.Embedding(sizes.tgt_vocab, sizes.emb),
        Generator(sizes.hidden, sizes.tgt_vocab, sizes),
        sizes
    )

    return model.cuda() if USE_CUDA else model


def run_epoch(data_iter: Iterable['Batch'], model: 'EncoderDecoder', loss_compute: 'SimpleLossCompute', print_every=50):
    start = time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        out, _, pre_output = model.forward(batch.src,
                                           batch.trg,
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


def greedy_decode(model: 'EncoderDecoder', src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence by choosing the token with maximum probability at each step"""
    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter, model: 'EncoderDecoder', n=2, max_len=100, sos_index=1, src_eos_index=None,
                   trg_eos_index=None, src_vocab=None, trg_vocab=None):
    model.eval()
    count = 0
    print()

    src_eos_index = None
    trg_sos_index = 1
    trg_eos_index = None

    for i, batch in enumerate(example_iter):
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> if present
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg

        result, _ = greedy_decode(
            model, batch.src, batch.src_mask, batch.src_lengths, max_len=max_len, sos_index=trg_sos_index,
            eos_index=trg_eos_index
        )

        print("Example #%d" % (i + 1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()

        count += 1
        if count == n:
            break


# *** classes ***
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1).long()
                              )
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


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
                 generator: 'Generator', sizes: 'Sizes'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.sizes = sizes

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor,
                src_lengths: torch.Tensor, trg_lengths: torch.Tensor):
        """Encode and decode the masked source and target sequences

        Parameters
        ----------
        src: torch.Tensor
            Batched input sequence embeddings, shape [batch size, sequence length]
        trg: torch.Tensor
            Batched target sequence embeddings, shape [batch size, sequence length]
        src_mask: torch.Tensor
            Boolean array of elements that are not padding (src)
        trg_mask: torch.Tensor
            Boolean array of elements that are not padding (trg)
        src_lengths: list
             A list of lengths of each src sequence (neglecting padding)
        trg_lengths: list
            A list of lengths of each trg sequence (neglecting padding)

        """
        encoder_hidden, encoder_final = self.encode(src, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_lengths):
        """See Encoder.forward for details"""
        # -1 because the source lacks the SOS token
        if self.training: assert src.shape == (self.sizes.batch, self.sizes.sequence_length - 1)
        return self.encoder(self.src_embed(src), src_lengths)

    def decode(self, encoder_hidden: torch.Tensor, encoder_final: torch.Tensor, src_mask: torch.Tensor,
               trg: torch.Tensor, trg_mask: torch.Tensor, decoder_hidden=None):
        """See Decoder.forward for details"""
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final, src_mask, trg_mask,
                            hidden=decoder_hidden)


class Generator(nn.Module):
    """Define standard linear + softmax generation step to create probabilities"""

    def __init__(self, hidden_size, vocab_size, sizes):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.sizes = sizes

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

    def __init__(self, input_size: int, hidden_size: int, sizes: 'Sizes', num_layers: int = 1, dropout: float = 0.):
        super().__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.sizes = sizes

    def forward(self, x, lengths):
        """Applies a bidirectional GRU to a sequence of embeddings

        Parameters
        ----------
        x:
            Padded input batch of variable length sequences. Must be sorted by length in decreasing order.
            Dimensions [batch, time, dim].
        lengths: torch.Tensor or list
            List of sequence lengths of each batch element. It is assumed that sequences are padded after
            iterating through the corresponding number of elements of the sequence.

        Returns
        -------
        output: torch.Tensor
            Hidden state (h_t) from the last layer of the GRU, for each t
        h_final: torch.Tensor
            Hidden state at the final step t=L, where the forwards and backwards hidden states of the
            bidirectional GRU are concatenated together
        """
        if self.training:
            assert x.shape == (self.sizes.batch, self.sizes.sequence_length - 1, self.sizes.emb)

        # Packed padded sequence allows us to pass mini-batches to an RNN, and have the RNN stop updating the hidden
        # state once we have reached the end of the sequence. Has greater GPU efficiency.
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        output, h_final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # inverse of pack_padded_sequence

        if self.training:
            # 2* because bidirectional, -1 because src strips out the first (SOS) character
            assert output.shape == (self.sizes.batch, self.sizes.sequence_length - 1, 2 * self.sizes.hidden)
            assert h_final.shape == (2 * self.num_layers, self.sizes.batch, self.sizes.hidden)

        # Concatenate the final states for both RNN directions. This is a summary of the entire sentence and will
        # be used as input to the decoder.
        fwd_final = h_final[0:h_final.size(0):2]
        bwd_final = h_final[1:h_final.size(0):2]
        h_final = torch.cat([fwd_final, bwd_final], dim=2)

        if self.training:
            assert h_final.shape == (1, self.sizes.batch, self.sizes.hidden * 2)

        return output, h_final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention TODO: conditional? Generally need to understand this

    Parameters
    ----------
    emb_size: int
        Dimension of the embedding space
    hidden_size: int
        Dimension of hidden state
    attention: BahdanauAttention
        An attention mechanism
    num_layers: int
        Number of GRU layers for decoder
    dropout: float
        Dropout probability
    bridge: bool, optional
        Whether to use a linear function to project the encoder final state down to a dimension of hidden_size.
    """

    def __init__(self, emb_size: int, hidden_size: int, attention: 'BahdanauAttention', sizes: 'Sizes',
                 num_layers: int = 1,
                 dropout: float = 0.5,
                 bridge: Optional[bool] = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.sizes = sizes

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embedded: torch.Tensor, encoder_hidden: torch.Tensor, src_mask: torch.Tensor,
                     proj_key: torch.Tensor, hidden: torch.Tensor):
        """Perform a single decoder step (1 word). This is used for inference.

        Parameters
        ----------
        prev_embedded: torch.Tensor
            Target embedding for the curent word to be decoded (teacher forcing), or the previous hidden state
            of the decoder (not teacher forcing)
        encoder_hidden:
            Hidden state (h_t) from the last layer of the encoder GRU, for each step in the input sequence t
        src_mask:
            Boolean array of elements that are not padding (src)
        proj_key:
            See BahdanauAttention.forward
        hidden:
            See `query` for BahdanauAttention.forward. This is the previous hidden state of the decoder.

        Returns
        -------
        output:
            The GRU state
        hidden:
            The GRU state, but a different shape
        pre_output:
            Hidden state vector, a combination of the prev_embedded, the decoder state, and the context

        """
        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [1, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            src_mask, query=query, proj_key=proj_key, value=encoder_hidden,
        )

        # update rnn hidden state
        rnn_input = torch.cat([prev_embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embedded, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        if not self.training:
            assert output.shape == (self.sizes.batch, 1, self.sizes.hidden)
            assert hidden.shape == (self.sizes.batch, 1, self.sizes.hidden)
            assert pre_output.shape == (self.sizes.batch, 1, self.sizes.hidden)
        return output, hidden, pre_output

    def forward(self, trg_embed: torch.Tensor, encoder_hidden: torch.Tensor, encoder_final: torch.Tensor,
                src_mask: torch.Tensor, trg_mask: torch.Tensor, hidden=None, max_len=None):
        """Unroll the decoder one step at a time. This is used for training.

        Parameters
        ----------
        trg_embed: torch.Tensor
            Target embedding. We use this for teacher forcing. At inference time, this becomes the embedding of the
            previous word. TODO: I find this questionable? Isnt this t-1 out of sync compared with training time?
        encoder_hidden: torch.Tensor
            Hidden state (h_t) from the last layer of the encoder GRU, for each step in the input sequence t
        encoder_final:
            Hidden state at the final step t=L, where the forwards and backwards hidden states of the
            bidirectional GRU are concatenated together
        src_mask:
            Boolean array of elements that are not padding (src)
        trg_mask:
            Boolean array of elements that are not padding (trg)
        hidden: torch.Tensor
            Decoder hidden state (s_i in Bahdanau)
        max_len: int, optional
            Maximum length of any sequence. If None, assumed to be the final dimension of trg_mask.

        """
        if self.training:
            assert trg_embed.shape == (self.sizes.batch, self.sizes.sequence_length - 1, self.sizes.emb)
            assert encoder_hidden.shape == (self.sizes.batch, self.sizes.sequence_length - 1, 2 * self.sizes.hidden)
            assert encoder_final.shape == (1, self.sizes.batch, self.sizes.hidden * 2)

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        if self.training:  assert hidden.shape == (1, self.sizes.batch, self.sizes.hidden)

        # Since U_a h_j (see Bahdanau) does not depend on the particular word being decoded, we can
        # pre-compute the projected encoder hidden states in advance to minimize
        # computational cost.
        # (the "keys" for the attention mechanism)
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


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention

    Parameters
    ----------
    sizes: Sizes
        A namedtuple of model sizes. Needs only hidden size
    key_size: int
        Size of the key hidden state for the matrix U_a in U_a h_j of the alignment model
    query_size:
        Size of the query hidden state for the matrix W_a s_{i-1} of the alignment model
    """

    def __init__(self, sizes: Sizes, key_size: Optional[int] = None, query_size: Optional[int] = None):
        super().__init__()
        self.sizes = sizes

        key_size = 2 * sizes.hidden if key_size is None else key_size  # assume a bi-directional encoder
        query_size = sizes.hidden if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, sizes.hidden, bias=False)
        self.query_layer = nn.Linear(query_size, sizes.hidden, bias=False)
        self.energy_layer = nn.Linear(sizes.hidden, 1, bias=False)

        self.alphas = None  # for storing attention probabilities

    def forward(self, mask: torch.Tensor, query: torch.Tensor, proj_key: torch.Tensor, value: torch.Tensor):
        """

        Parameters
        ----------
        mask: torch.Tensor
            Boolean tensor of non-padding elements of sequences
        query: torch.Tensor
            The previous hidden state of the decoder, s_{i-1}
        proj_key: torch.Tensor
            Hidden state, for each step in the input sequence t. This has been projected via a linear transformation
            from 2 * hidden_size, down to hidden, in the final dimension, U_a h_j in Bahdanau. These are the "keys" of
            an attention layer h_j in Bahdanau.
        value: torch.Tensor
            Hidden state (h_t) from the last layer of the encoder GRU, for each step in the input sequence t. The object
            to be reweighted in calculating the context.

        Returns
        -------
        context: torch.Tensor
            A vector, which uses an attention mechanism to reweight the hidden states (values)
        alphas: torch.Tensor
            Probability that target word i is aligned to, or translated from, source word j

        """

        if self.training:
            assert query.shape == (self.sizes.batch, 1, self.sizes.hidden)
            assert proj_key.shape == (self.sizes.batch, self.sizes.sequence_length - 1, self.sizes.hidden)
            assert value.shape == (self.sizes.batch, self.sizes.sequence_length - 1, 2 * self.sizes.hidden)
            assert mask.shape == (self.sizes.batch, 1, self.sizes.sequence_length - 1)

        # Project the query (decoder state), W_a s_{i-1}.
        query = self.query_layer(query)

        # Calculate attention energies (scores) via an alignment model
        # e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_z h_j)
        scores = self.energy_layer(torch.tanh(query + proj_key))
        if self.training: assert scores.shape == (self.sizes.batch, self.sizes.sequence_length - 1, 1)

        scores = scores.squeeze(2).unsqueeze(1)
        if self.training: assert scores.shape == (self.sizes.batch, 1, self.sizes.sequence_length - 1)

        #  Mask invalid positions
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        # alpha_ij can be interpreted as the probability that target word i is aligned to, or translated from,
        # source word j
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas
        if self.training: assert alphas.shape == (self.sizes.batch, 1, self.sizes.sequence_length - 1)

        # compute weighted sum of the "annotations" of every step of the input
        context = torch.bmm(alphas, value)

        if self.training:
            assert context.shape == (self.sizes.batch, 1, 2 * self.sizes.hidden)
            assert alphas.shape == (self.sizes.batch, 1, self.sizes.sequence_length - 1)

        return context, alphas
