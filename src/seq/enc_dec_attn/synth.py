"""
Random symbol generator for source:target memorization task. Run this module as a script to perform training/inference.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from seq import USE_CUDA
from seq.enc_dec_attn.model import make_model, run_epoch, SimpleLossCompute, print_examples, Sizes
from seq.enc_dec_attn.parse import Batch


def data_gen(vocab_size, batch_size, num_batches, length, pad_index=0, sos_index=1):
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


def train_copy_task():
    """Train the simple copy task."""
    vocab_size = 11
    emb_size = 32
    hidden_size = 64
    batch_size = 32
    num_batches = 100
    num_layers = 1
    dropout = 0.1
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    sequence_length = 10

    sizes_train = Sizes(src_vocab=vocab_size,
                        tgt_vocab=vocab_size,
                        emb=emb_size,
                        hidden=hidden_size,
                        num_layers=num_layers,
                        batch=batch_size,
                        num_batches=num_batches,
                        sequence_length=sequence_length
                        )

    model = make_model(sizes_train, dropout=dropout)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(vocab_size, 1, num_batches, sequence_length))

    dev_perplexities = []

    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(vocab_size, batch_size, sizes_train.num_batches, sequence_length)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad():
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=9)

    return dev_perplexities


def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)


def main():
    dev_perplexities = train_copy_task()
    plot_perplexity(dev_perplexities)


if __name__ == '__main__':
    # In VS Code, before running
    # cd sequence-models/src
    # export PYTHONPATH=$(pwd)
    # https://code.visualstudio.com/docs/python/environments#_use-of-the-pythonpath-variable
    main()
