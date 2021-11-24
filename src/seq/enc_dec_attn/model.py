import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.

    Parameters
    ----------
    encoder: nn.Module
        Encodes a batched sequence of padded word embeddings, returning output for each input sequence and
        final hidden state (concatenation of forwards and backwards final RNN states)

    decoder: nn.Module

    src_embed: nn.Module
        Embedding layer for the source

    trg_embed: nn.Module
        Embedding layer for the target

    generator: nn.Module
        Layer to consume matrix of hidden states and output probabilities over the vocabulary TODO: Check

    """

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths):
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

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
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
    """A conditional RNN decoder with attention TODO: conditional?

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

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embedded, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)

        Parameters
        ----------
        prev_embedded
        encoder_hidden
        src_mask
        proj_key
        hidden

        Returns
        -------

        """



