# encoding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import match_rnn
from modules.layers import pointer
from modules.layers import choose
import utils


class Model(nn.Module):
    """ match-lstm model for machine comprehension"""
    def __init__(self, param):
        """
        :param param: embedding, hidden_size, dropout_p, encoder_dropout_p, encoder_direction_num, encoder_layer_num
        """
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder
        input_size = self.embedding.embedding_dim - 9
        self.encoder_a = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=False
        )

        input_size = self.embedding.embedding_dim
        self.encoder_p_q = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=True
        )

        # match rnn
        input_size = self.hidden_size * 2
        self.match_rnn = match_rnn.MatchRNN(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=True,
            is_bn=self.is_bn
        )

        # addition_rnn
        input_size = self.hidden_size * 2
        self.addition_rnn = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )

        # mean passage based on attn
        self.mean_p = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size*2
        )

        # outputs
        self.choose = choose.Choose(self.hidden_size*2, self.hidden_size*2)

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range (2, batch_size, content_len)
        """
        passage = batch[: 3]
        query = batch[3: 6]
        alter_1 = batch[6]
        alter_2 = batch[7]
        alter_3 = batch[8]

        # mask
        passage_mask = utils.get_mask(passage[0])
        query_mask = utils.get_mask(query[0])
        alter1_mask = utils.get_mask(alter_1)
        alter2_mask = utils.get_mask(alter_2)
        alter3_mask = utils.get_mask(alter_3)

        # embedding
        passage_vec = self.embedding(passage)
        query_vec = self.embedding(query)
        alter1_vec = self.embedding(alter_1)
        alter2_vec = self.embedding(alter_2)
        alter3_vec = self.embedding(alter_3)

        # encode
        passage_vec = self.encoder_p_q(passage_vec, passage_mask)
        query_vec = self.encoder_p_q(query_vec, query_mask)

        # encoder alters
        # 均值
        alter1_vec = self.encoder_a(alter1_vec, alter1_mask)  # (a_len, batch_size, hidden_size*2)
        alter1_vec = utils.compute_mean(alter1_vec, alter1_mask)
        alter2_vec = self.encoder_a(alter2_vec, alter2_mask)
        alter2_vec = utils.compute_mean(alter2_vec, alter2_mask)
        alter3_vec = self.encoder_a(alter3_vec, alter3_mask)
        alter3_vec = utils.compute_mean(alter3_vec, alter3_mask)

        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec])
        alters = alters.transpose(0, 1)  # (batch_size, 3, hidden_size*2)

        # match rnn
        hr = self.match_rnn(passage_vec, passage_mask, query_vec, query_mask)

        # aggregation
        hr = self.addition_rnn(hr, passage_mask)

        # mean p
        hr = self.mean_p(hr, passage_mask)  # (batch_size, hidden_size*2)

        # outputs
        outputs = self.choose(hr, alters)
        outputs = f.log_softmax(outputs, dim=1)

        return outputs
