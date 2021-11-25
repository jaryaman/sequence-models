"""
Random symbol generator for source:target memorization task
"""

import torch
import numpy as np
from seq import USE_CUDA
from seq.enc_dec_attn.parse import Batch


def data_gen(vocab_size=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task"""
    for i in range(num_batches):
        data = torch.from_numpy(
            np.random.randint(1, vocab_size, size=(batch_size, length))
        )
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length - 1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
