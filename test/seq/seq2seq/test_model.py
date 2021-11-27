import pytest
import torch
import torch.nn as nn
from torch import optim
import random
from seq.seq2seq.model import (SOS_token, EncoderRNN, DecoderRNN, AttnDecoderRNN, Lang, tensor_from_sentence,
                               train, tensors_from_pair)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope='session')
def make_langs():
    pairs = [['j ai ans .', 'i m .'],
             ['je vais bien .', 'i m ok .'],
             ['ca va .', 'i m ok .']]
    input_lang = Lang('fra')
    output_lang = Lang('eng')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return pairs, input_lang, output_lang


class TestLang:
    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_lang(self, make_langs):
        self.prepare_data(make_langs)
        assert self.input_lang.name == 'fra'
        assert self.input_lang.n_words == 11
    
    def test_foo():
        assert False


class TestEncoderRNN:
    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_encoder(self, make_langs):
        self.prepare_data(make_langs)

        n_hidden = 5
        encoder = EncoderRNN(self.input_lang.n_words, n_hidden, DEVICE)
        a_sentence = tensor_from_sentence(self.input_lang, self.pairs[0][0], DEVICE)

        hidden = encoder.init_hidden()
        for i in range(len(a_sentence)):
            output, hidden = encoder(a_sentence[i], hidden)

        assert output.size() == (1, 1, n_hidden)
        assert hidden.size() == (1, 1, n_hidden)

class TestDecoderRNN:
    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_decoder(self, make_langs):
        self.prepare_data(make_langs)

        n_hidden = 5
        decoder = DecoderRNN(n_hidden, self.output_lang.n_words, DEVICE)


        decoder_input = torch.tensor([SOS_token], device=DEVICE)
        target_length = 6
        hidden = decoder.init_hidden()

        for i in range(target_length):
            decoder_output, hidden = decoder(decoder_input, hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()

        assert decoder_output.size() == (1, self.output_lang.n_words)
        assert hidden.size() == (1, 1, n_hidden)

class TestAttnDecoderRNN:
    n_hidden = 5
    max_length = 11

    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_attn_decoder_without_teacher_forcing(self, make_langs):
        self.prepare_data(make_langs)

        n_hidden = 5
        attn_decoder = AttnDecoderRNN(self.n_hidden,
                                      self.output_lang.n_words,
                                      self.max_length,
                                      DEVICE)

        decoder_input = torch.tensor([SOS_token], device=DEVICE)
        target_length = 6
        hidden = attn_decoder.init_hidden()

        encoder_outputs = torch.zeros(self.max_length, self.n_hidden, device=DEVICE)

        for i in range(target_length):
            decoder_output, hidden, decoder_attention = attn_decoder(decoder_input, hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()

        assert decoder_output.size() == (1, self.output_lang.n_words)
        assert hidden.size() == (1, 1, n_hidden)
        assert decoder_attention.size() == (1, self.max_length)

class TestTrain:
    n_hidden = 5
    max_length = 11
    learning_rate=0.01
    n_iters = 10


    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def create_encoders(self, make_langs):
        self.prepare_data(make_langs)
        encoder = EncoderRNN(self.input_lang.n_words, self.n_hidden, DEVICE)
        attn_decoder = AttnDecoderRNN(self.n_hidden,
                                      self.output_lang.n_words,
                                      self.max_length,
                                      DEVICE)
        return encoder, attn_decoder

    def test_train_without_teacher_forcing(self, make_langs):
        encoder, decoder = self.create_encoders(make_langs)
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=self.learning_rate)
        input_tensor, target_tensor = tensors_from_pair(self.pairs[0], self.input_lang, self.output_lang, DEVICE)
        criterion = nn.NLLLoss()

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, 0,
                     self.max_length,
                     DEVICE)
        assert isinstance(loss, float)