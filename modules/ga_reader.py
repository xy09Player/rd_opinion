# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
import utils
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import pointer
from modules.layers import choose


class Model(nn.Module):
    """ ga reader, paper: Gated-attention readers for text comprehension """

    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']
        self.k = 2

        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder_a
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

        # encoder p, q
        self.doc_list = nn.ModuleList()
        self.query_list = nn.ModuleList()
        input_size = self.embedding.embedding_dim
        for i in range(self.k):
            di_enc = encoder.Rnn(
                mode=self.mode,
                input_size=input_size if i == 0 else self.hidden_size*2,
                hidden_size=self.hidden_size,
                dropout_p=self.dropout_p,
                bidirectional=True,
                layer_num=1,
                is_bn=self.is_bn
            )

            qi_enc = encoder.Rnn(
                mode=self.mode,
                input_size=input_size,
                hidden_size=self.hidden_size,
                dropout_p=self.dropout_p,
                bidirectional=True,
                layer_num=1,
                is_bn=self.is_bn
            )
            self.doc_list.append(di_enc)
            self.query_list.append(qi_enc)

        # mean passage based on attn
        self.mean_p = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size*2
        )

        # mean answer based on attn
        self.mean_a = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size*2
        )

        # outputs
        self.choose = choose.Choose(self.hidden_size*2, self.hidden_size*2)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, batch):

        passage = batch[:3]
        query = batch[3: 6]
        alter1 = batch[6]
        alter2 = batch[7]
        alter3 = batch[8]

        # mask
        passage_mask = utils.get_mask(passage[0])
        query_mask = utils.get_mask(query[0])
        alter1_mask = utils.get_mask(alter1)
        alter2_mask = utils.get_mask(alter2)
        alter3_mask = utils.get_mask(alter3)

        # embedding
        passage_vec = self.embedding(passage)
        query_vec = self.embedding(query)
        alter1_vec = self.embedding(alter1)
        alter2_vec = self.embedding(alter2)
        alter3_vec = self.embedding(alter3)

        # encoder alters
        alter1_vec = self.encoder_a(alter1_vec, alter1_mask)  # (a_len, batch_size, hidden_size*2)
        alter1_vec = self.mean_a(alter1_vec, alter1_mask)  # (batch_size, hidden_size*2)
        alter2_vec = self.encoder_a(alter2_vec, alter2_mask)
        alter2_vec = self.mean_a(alter2_vec, alter2_mask)
        alter3_vec = self.encoder_a(alter3_vec, alter3_mask)
        alter3_vec = self.mean_a(alter3_vec, alter3_mask)
        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec])
        alters = alters.transpose(0, 1)  # (batch_size, 3, hidden_size*2)

        # fusion: interation, attn
        for i in range(self.k):
            di_rnn = self.doc_list[i]
            qi_rnn = self.query_list[i]

            passage_vec = di_rnn(passage_vec, passage_mask)  # (p_len, batch_size, hidden_size*2)
            query_vec_tmp = qi_rnn(query_vec, query_mask)  # (q_len, batch_size, hidden_size*2)

            # ga
            passage_vec = passage_vec.transpose(0, 1)  # (batch_size, p_len, h*2)
            query_vec_tmp = query_vec_tmp.transpose(0, 1)  # (batch_size, q_len, h*2)
            alpha = torch.bmm(passage_vec, query_vec_tmp.transpose(1, 2))  # (batch_size, p_len, q_len)
            mask = query_mask.eq(0).unsqueeze(1).expand(alpha.size())  # (batch_size, p_len, q_len)
            alpha.masked_fill_(mask, -float('inf'))
            alpha = f.softmax(alpha, dim=2)  # (batch_size, p_len, q_len)
            query_vec_tmp = torch.bmm(alpha, query_vec_tmp)  # (batch_size, p_len, h*2)
            passage_vec = passage_vec * query_vec_tmp

            passage_vec = self.dropout(passage_vec)  # (batch_size, p_len, h*2)
            passage_vec = passage_vec.transpose(0, 1)  # (p_len, batch_size, h*2)

        # outputs
        passage_vec = self.mean_p(passage_vec, passage_mask)  # (batch_size, h*2)
        outputs = self.choose(passage_vec, alters)
        outputs = f.log_softmax(outputs, dim=1)

        return outputs
