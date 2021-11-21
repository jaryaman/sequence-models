import pytest
import torch

from seq.seq2seq.model import EncoderRNN, DecoderRNN, Lang, tensor_from_sentence

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


class TestEncoderRNN:
    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_encoder(self, make_langs):
        self.prepare_data(make_langs)

        n_hidden = 5
        encoder = EncoderRNN(self.input_lang.n_words, n_hidden, DEVICE)
        a_sentence = tensor_from_sentence(self.input_lang, self.pairs[0][0], DEVICE)

        h0 = encoder.init_hidden()
        output, h_n = encoder(a_sentence, h0)
        assert output.size() == (5, 1, n_hidden)
        assert h_n.size() == (1, 1, n_hidden)

class TestDecoderRNN:
    def prepare_data(self, make_langs):
        self.pairs, self.input_lang, self.output_lang = make_langs

    def test_decoder(self, make_langs):
        self.prepare_data(make_langs)

        n_hidden = 5
        decoder = DecoderRNN(n_hidden, self.output_lang.n_words, DEVICE)
        a_sentence = tensor_from_sentence(self.output_lang, self.pairs[0][1], DEVICE)

        h0 = decoder.init_hidden()
        output, h_n = decoder(a_sentence, h0)
        assert output.size() == (len(a_sentence), self.output_lang.n_words)
        assert h_n.size() == (1, 1, n_hidden)