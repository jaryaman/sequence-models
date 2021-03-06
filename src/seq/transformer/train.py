import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from seq.transformer.model import EncoderDecoder, Generator


# *** functions ***
def run_epoch(data_iter: Iterable['Batch'], model: EncoderDecoder, loss_compute: 'SimpleLossCompute'):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i}, Loss: {loss / batch.ntokens}, Tokens per Sec: {tokens / elapsed}")
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def subsequent_mask(size):
    """Mask out subsequent positions. This blocks attending to future words during training."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


# *** classes ***
class Batch:
    """Object for holding a batch of data with mask during training"""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # This goes into the decoder for training. At each step, decoder can see current token, and all previous
            # tokens. These are the t-1 tokens, because we're not allowed to see the true token at training time,
            # hence :-1
            self.tgt = tgt[:, :-1]

            # Trying to predict this: the next token. This is what goes into the loss. We don't try to predict the
            # starter token, hence 1:
            self.tgt_y = tgt[:, 1:]

            # This mask enforces the model not being able to peek at the true, or future, target tokens in the decoder.
            self.tgt_mask = self.make_non_anticipating_tgt_mask(self.tgt, pad)

            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_non_anticipating_tgt_mask(tgt, pad):
        """Create a mask to hide padding and future words, to preserve the autoregressive property of the decoder"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class NoamOpt:
    """Learning rate schedule"""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return (self.factor *
                (self.model_size ** (-0.5) *
                 min(step ** (-0.5), step * self.warmup ** (-1.5))
                 )
                )


class LabelSmoothing(nn.Module):
    """Implement label smoothing

     See https://arxiv.org/abs/1512.00567

     Parameters
     ----------
     size: int
        The total number of categories. In language models, this is the vocabulary size. Used to define the prior.
     padding_idx: int
        The token used for padding
     smoothing: float
        The amount of label smoothing. If 0, labels are retained as binary. If 1, then all information from the label
        discarded, and the prior is used for labels. Between 0-1.

     Notes
     -----
     Uses KL divergence loss
     """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.size(1) == self.size
        assert 0. <= self.smoothing <= 1.
        true_dist = torch.zeros_like(x)

        non_padding_words_in_vocab = self.size - 1
        true_dist.fill_(self.smoothing / (non_padding_words_in_vocab - 1))  # -1 to leave space for the true label
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # insert the true labels as 1-prior
        true_dist[:, self.padding_idx] = 0  # the padding is never a true class

        # assert torch.allclose(true_dist.sum(dim=1), torch.tensor(1.))

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    def __init__(self, generator: 'Generator', criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm):
        x = self.generator(x)  # Final linear and softmax, after the RHS box of Fig 1 from Attention is All You Need
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data.item() * norm
