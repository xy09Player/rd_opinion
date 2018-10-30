# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
import utils
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import choose


class Model(nn.Module):
    """
     Stanfor AR, paper "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"
    """

    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder_p: 双向rnn
        input_size = self.embedding.embedding_dim
        self.encoder_p = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=True
        )

        # encoder_q: 单向rnn
        self.encoder_q = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=False,
            layer_num=self.encoder_layer_num,
            is_bn=True
        )

        # encoder_a: 单向rnn
        input_size = self.embedding.sd_embedding.embedding_dim
        self.encoder_a = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=False,
            layer_num=self.encoder_layer_num,
            is_bn=False
        )

        # similar W
        self.sim_w = nn.Linear(self.hidden_size, self.hidden_size*2)

        # outputs
        self.choose = choose.Choose(self.hidden_size*2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, batch):
        """
        :param batch:
        :return: (batch_size, 3)
        """
        passage = batch[:3]
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

        # encoder
        passage_vec = self.encoder_p(passage_vec, passage_mask)  # (c_len, batch_size, hidden_size*2)
        _, query_vec = self.encoder_q(query_vec, query_mask, need_final_state=True)  # (batch_size, hidden_size)
        _, alter1_vec = self.encoder_a(alter1_vec, alter1_mask, need_final_state=True)  # (batch_size, hidden_size)
        _, alter2_vec = self.encoder_a(alter2_vec, alter2_mask, need_final_state=True)
        _, alter3_vec = self.encoder_a(alter3_vec, alter3_mask, need_final_state=True)
        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec]).transpose(0, 1)  # (batch_size, 3, hidden_size)

        # attention
        alpha = self.sim_w(query_vec).unsqueeze(2)  # (batch_size, hidden_size*2, 1)
        alpha = torch.bmm(passage_vec.transpose(0, 1), alpha).squeeze(2)  # (batch_size, c_len)
        mask = passage_mask.eq(0)
        alpha.masked_fill_(mask, -float('inf'))
        alpha = f.softmax(alpha, dim=1).unsqueeze(1)  # (batch_size, 1, c_len)
        passage_vec = torch.bmm(alpha, passage_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*2)

        # outputs
        outputs = self.choose(passage_vec, alters)
        outputs = f.log_softmax(outputs, dim=1)

        return outputs






