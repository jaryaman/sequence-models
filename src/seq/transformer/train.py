import time

import numpy as np
import torch
import torch.nn as nn

from seq.transformer.model import EncoderDecoder


# *** functions ***
def run_epoch(data_iter, model: EncoderDecoder, loss_compute: 'SimpleLossCompute'):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
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

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
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
        true_dist.fill_(self.smoothing / (self.size - 2))  # baseline prior
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # insert the true labels as 1-prior
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data.item() * norm
