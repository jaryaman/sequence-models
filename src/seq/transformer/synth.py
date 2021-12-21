"""A copy task on synthetic data using a transformer"""

import numpy as np
import torch

from seq.transformer.model import make_model, Sizes
from seq.transformer.train import NoamOpt, Batch, LabelSmoothing, run_epoch, SimpleLossCompute, subsequent_mask


def data_gen(vocab_size, sequence_length, batch, n_batches):
    """Generate random data for a src-tgt copy task"""
    for i in range(n_batches):
        data = np.random.randint(1, vocab_size, size=(batch, sequence_length), dtype='int64')
        data[:, 0] = 1
        src = torch.from_numpy(data)
        tgt = torch.from_numpy(data)
        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def decode_on_sequential_source(model, sizes: 'Sizes'):
    model.eval()
    with torch.no_grad():
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5]])
        assert src.shape[-1] == sizes.src_seq
        src_mask = torch.ones(1, 1, sizes.src_seq)
        decoded = greedy_decode(model, src, src_mask, max_len=sizes.tgt_seq, start_symbol=1)
    return decoded


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    import random
    random.seed(0)
    torch.use_deterministic_algorithms(True)

    vocab_size = 10 + 1  # padding is included in the vocabulary, so +1
    seq_len = 15
    sizes = Sizes(
        batch=30,
        n_batches=20,
        src_seq=seq_len,
        tgt_seq=seq_len,
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        n_layers=2,
        emb=512,
        d_ff=2048,
        h=8,
        )

    criterion = LabelSmoothing(size=sizes.tgt_vocab, padding_idx=0, smoothing=0.1)
    model = make_model(sizes)
    model_opt = NoamOpt(sizes.emb, factor=1, warmup=400,
                        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # training
    for epoch in range(10):
        model.train()
        print(f'Epoch: {epoch}')
        run_epoch(
            data_gen(sizes.tgt_vocab, sizes.src_seq, sizes.batch, sizes.n_batches),
            model,
            SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        with torch.no_grad():
            print('Evaluation on random sequence. ')
            loss_normed = run_epoch(
                data_gen(sizes.tgt_vocab, sizes.src_seq, sizes.batch, 5),
                model,
                SimpleLossCompute(model.generator, criterion, None))
            print(f'Evaluation loss: {loss_normed}')

        decoded = decode_on_sequential_source(model, sizes)
        print(f'Decoded on sequential source: {decoded}')
        print('=========================================')

    # Evaluation
    print('Final evaluation')
    decoded = decode_on_sequential_source(model, sizes)
    print(f'Decoded on sequential source: {decoded}')
    print('Done.')



if __name__ == '__main__':
    main()
