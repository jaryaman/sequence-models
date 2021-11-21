from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# *** global variables ***
SOS_token = 0  # start of sentence
EOS_token = 1  # end of sentence


# *** functions ***
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pairs(pair, input_lang, output_lang, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor


# *** classes ***
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    def __init__(self, in_vocab_size, hidden_size, device):
        super().__init__()

        self.embedding = nn.Embedding(in_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        Sizes = namedtuple('Size', ['hidden', 'in_vocab'])
        self.sizes = Sizes(hidden=hidden_size, in_vocab=in_vocab_size)
        self.device = device

    def forward(self, x_in, hidden):
        sequence_length, batch_size = x_in.size()

        embedded = self.embedding(x_in)  # .view(1, 1, -1)

        assert embedded.size() == (sequence_length, batch_size, self.sizes.hidden)

        output = embedded
        output, hidden = self.gru(output, hidden)

        assert output.size() == (sequence_length, batch_size, self.sizes.hidden)
        assert hidden.size() == (1, batch_size, self.sizes.hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros((1, 1, self.sizes.hidden), device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, out_vocab_size, device):
        super().__init__()
        self.embedding = nn.Embedding(out_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        Sizes = namedtuple('Size', ['hidden', 'out_vocab'])
        self.sizes = Sizes(hidden=hidden_size, out_vocab=out_vocab_size)
        self.device = device

    def forward(self, x_in, hidden):
        sequence_length, batch_size = x_in.size()

        output = self.embedding(x_in)
        assert output.size() == (sequence_length, batch_size, self.sizes.hidden)

        output = F.relu(output)
        assert output.size() == (sequence_length, batch_size, self.sizes.hidden)

        output, hidden = self.gru(output, hidden)
        assert output.size() == (sequence_length, batch_size, self.sizes.hidden)

        output = self.softmax(self.out(torch.squeeze(output, 1)))

        assert output.size() == (sequence_length, self.sizes.out_vocab)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.sizes.hidden, device=self.device)
