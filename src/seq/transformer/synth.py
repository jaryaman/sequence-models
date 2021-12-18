import numpy as np
import torch

from seq.transformer.model import make_model
from seq.transformer.train import NoamOpt, Batch, LabelSmoothing, run_epoch, SimpleLossCompute, subsequent_mask


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task"""
    for i in range(nbatches):
        data = np.random.randint(1, V, size=(batch, 10), dtype='int64')
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

def decode_on_sequential_source(model):
    model.eval()
    with torch.no_grad():
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        src_mask = torch.ones(1, 1, 10)
        print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


def main():
    np.random.seed(42)
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # training
    for epoch in range(100):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        with torch.no_grad():
            print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

        decode_on_sequential_source(model)

    # Evaluation
    print('Final evaluation')
    decode_on_sequential_source(model)



if __name__ == '__main__':
    main()
