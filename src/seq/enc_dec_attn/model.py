import torch.nn as nn
import torch.nn.functional as F
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.

    Parameters
    ----------
    encoder:

    decoder:

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
        """Encode and decode the masked source and target sequences"""
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
