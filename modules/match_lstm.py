# encoding = utf-8
# author = xy

from torch import nn
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import match_rnn
from modules.layers import pointer
import utils


class Model(nn.Module):
    """ match-lstm model for machine comprehension"""
    def __init__(self, param):
        """
        :param param: embedding, hidden_size, dropout_p, encoder_dropout_p, encoder_direction_num, encoder_layer_num
        """
        super(Model, self).__init__()

        self.w2v_size = param['embedding'].shape[1]
        self.vocab_size = param['embedding'].shape[0]
        self.embedding_type = param['embedding_type']
        self.embedding_is_training = param['embedding_is_training']
        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_bidirectional = param['encoder_bidirectional']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        if self.embedding_type == 'standard':
            self.embedding = embedding.Embedding(param['embedding'])
            is_bn = False
        else:
            self.embedding = embedding.ExtendEmbedding(param['embedding'])
            is_bn = True

        # encoder
        input_size = self.embedding.embedding_dim
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=self.encoder_bidirectional,
            layer_num=self.encoder_layer_num,
            is_bn=is_bn
        )

        # match rnn
        input_size = self.hidden_size * 2 if self.encoder_bidirectional else self.hidden_size
        self.match_rnn = match_rnn.MatchRNN(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=False,
            is_bn=self.is_bn
        )

        # pointer
        self.pointer_net = pointer.BoundaryPointer(
            mode=self.mode,
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            is_bn=self.is_bn
        )

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range (2, batch_size, content_len)
        """
        content = batch[: 3]
        question = batch[3: 6]

        # mask
        content_mask = utils.get_mask(content[0])  # (batch_size, seq_len)
        question_mask = utils.get_mask(question[0])

        # embedding
        content_vec = self.embedding(content)  # (seq_len, batch_size, embedding_dim)
        question_vec = self.embedding(question)

        # encoder
        content_vec = self.encoder(content_vec, content_mask)  # (seq_len, batch_size, hidden_size(*2))
        question_vec = self.encoder(question_vec, question_mask)

        # match-rnn
        hr = self.match_rnn(content_vec, content_mask, question_vec, question_mask)  # (p_seq_len, batch_size, hidden_size(*2))

        # pointer
        ans_range = self.pointer_net(hr, content_mask)

        return ans_range
