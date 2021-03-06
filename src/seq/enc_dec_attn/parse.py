from typing import Tuple

import torch

from seq import USE_CUDA


class Batch:
    """Object for holding source and target sentences, and their lengths and masks

    Parameters
    ----------
    src_tup: Tuple[torch.Tensor, list]
        Tuple of sequences and corresponding sequence lengths, for source sequences
    trg_tup: Tuple[torch.Tensor, list]
        Tuple of sequences and corresponding sequence lengths, for target sequences
    pad_index: int
        Character used for padding

    Attributes
    ----------
    src: torch.Tensor
        A batch of source sequences from a torch text iterator
    src_lengths: list
        A list of lengths of each sequence (neglecting padding)
    src_mask: torch.Tensor
        Boolean tensor of sequence elements that are not padding (source)
    nseqs: int
        Number of sequences in src
    trg: torch.Tensor
        A batch of target sequences from a torch text iterator, corresponding to src
    """

    def __init__(self, src_tup: Tuple[torch.Tensor, list], trg_tup: Tuple[torch.Tensor, list], pad_index=0):
        src, src_lengths = src_tup

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg_tup is not None:
            trg, trg_lengths = trg_tup
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg_tup is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()
