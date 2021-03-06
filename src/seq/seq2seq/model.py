"""
I abandoned this module when I realised that it's not quite the same model as the original Bahdanau paper
"""
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

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


def tensors_from_pair(pair, input_lang, output_lang, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor


def train(input_tensor,
          target_tensor,
          encoder: 'EncoderRNN',
          decoder: 'AttnDecoderRNN',
          encoder_optimizer,
          decoder_optimizer,
          criterion: nn.NLLLoss,
          teacher_forcing_ratio,
          max_length,
          device,
          ):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.sizes.hidden, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input: todo: ?

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(pairs,
                encoder,
                decoder,
                input_lang,
                output_lang,
                n_iters,
                device,
                teacher_forcing_ratio,
                max_length,
                learning_rate=0.01):

    losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()


    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio,
                     max_length,
                     device, )
        losses.append(loss)

    return losses


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

    def forward(self, encoder_input: torch.Tensor, hidden: torch.Tensor):
        """Step forward the encoder

        Parameters
        ----------
        encoder_input: torch.Tensor
            A single tokenized word
        hidden: torch.Tensor
            The previous hidden state, shape == [1, 1, hidden_size]

        Returns
        -------
        output: torch.Tensor
            RNN output for the single input word, shape == [1, 1, hidden_size]
        hidden: torch.Tensor
            RNN hidden state after evaluation of the RNN
        """
        embedded = self.embedding(encoder_input).view(1, 1, -1)  # sequence_length=1, batch_size=1, hidden_size

        assert embedded.size() == (1, 1, self.sizes.hidden)

        output = embedded
        output, hidden = self.gru(output, hidden)

        assert output.size() == (1, 1, self.sizes.hidden)
        assert hidden.size() == (1, 1, self.sizes.hidden)

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
        assert x_in.size() == (1,)

        output = self.embedding(x_in).view(1, 1, -1)
        assert output.size() == (1, 1, self.sizes.hidden)

        output = F.relu(output)
        assert output.size() == (1, 1, self.sizes.hidden)

        output, hidden = self.gru(output, hidden)
        assert output.size() == (1, 1, self.sizes.hidden)

        output = self.softmax(self.out(output[0]))

        assert output.size() == (1, self.sizes.out_vocab)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.sizes.hidden, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, out_vocab_size, max_length, device, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(out_vocab_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)  # alignment model
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_vocab_size)

        Sizes = namedtuple('Size', ['hidden', 'out_vocab', 'max_seq_length'])
        self.sizes = Sizes(hidden=hidden_size, out_vocab=out_vocab_size, max_seq_length=max_length)

        self.device = device

    def forward(self, decoder_input, hidden, encoder_outputs):
        """

        Parameters
        ----------
        decoder_input: torch.Tensor
            A single tokenized word. For the first step, this is SOS_token. Thereafter, it is the previous output
            of the decoder.
        hidden: torch.Tensor
            For the first step, this is the final hidden state of the encoder. Thereafter, it is the previous decoder
            hidden state.
        encoder_outputs: torch.Tensor
            The outputs of the encoder for every word of the sequence. Zero-padded to be max_length x n_hidden.

        Returns
        -------
        output:

        hidden:

        attn_weights:

        """
        assert encoder_outputs.size() == (self.sizes.max_seq_length, self.sizes.hidden)

        embedded = self.embedding(decoder_input).view(1, 1, -1)
        assert embedded.size() == (1, 1, self.sizes.hidden)
        assert hidden.size() == (1, 1, self.sizes.hidden)

        embedded = self.dropout(embedded)

        # In Bahdanau, they use the concatenation of the forwards and backwards hidden states of the whole encoder,
        # whereas this tutorial still uses the hidden state to feed the memory forwards. I suppose this makes it
        # a smaller/faster model to train.
        word_annotation = torch.cat((embedded[0], hidden[0]), dim=1)

        # e_ij in Bahdanau, alignment model. For each word to be decoded, generate a vector which scores the positions
        # on the encoder_outputs
        alignment_model = self.attn(word_annotation)
        assert alignment_model.size() == (1, self.sizes.max_seq_length)

        # alpha_ij in Bahdanau. Softmax makes a sparse scoring of the encoder_outputs
        attn_weights = F.softmax(alignment_model, dim=1)
        assert attn_weights.size() == (1, self.sizes.max_seq_length)

        # c_i in Bahdanau, context vector for word i in the decoder output
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        assert attn_applied.shape == (1, 1, self.sizes.hidden)

        output = torch.cat((embedded[0], attn_applied[0]), dim=1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.sizes.hidden, device=self.device)
