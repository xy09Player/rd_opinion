# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import pointer
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import choose
import utils


class Model(nn.Module):
    """ bi-rdf for reading comprehension """

    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder: p
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
        # encoder: q
        self.encoder_q = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=True
        )
        # encoder: a
        input_size = self.embedding.embedding_dim - 9
        self.encoder_a = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size//2,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=False
        )

        # self-att: a
        self.a_att = nn.Linear(self.hidden_size, 1, bias=False)

        # attention flow layer
        self.att_c = nn.Linear(self.hidden_size * 2, 1)
        self.att_q = nn.Linear(self.hidden_size * 2, 1)
        self.att_cq = nn.Linear(self.hidden_size * 2, 1)

        # modeling layer
        self.modeling_rnn = encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size * 8,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            layer_num=2,
            is_bn=self.is_bn
        )

        # prediction
        self.wq = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vq = nn.Linear(self.hidden_size, 1, bias=False)
        self.wp1 = nn.Linear(self.hidden_size*10, self.hidden_size, bias=False)
        self.wp2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vp = nn.Linear(self.hidden_size, 1, bias=False)
        self.predict = nn.Linear(self.hidden_size*10, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(self.dropout_p)

        self.reset_param()

    def reset_param(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: result (batch_size, 3)
        """

        def att_flow_layer(c, c_mask, q, q_mask):
            """
            attention flow layer
            :param c: (c_len, batch_size, hidden_size*2)
            :param c_mask: (batch_size, c_len)
            :param q: (q_len, batch_size, hidden_size*2)
            :param q_mask: (batch_size, q_len)
            :return: g (c_len, batch_size, hidden_size*8)
            """
            c_len = c.size(0)
            q_len = q.size(0)
            batch_size = c.size(1)

            c = c.transpose(0, 1)
            q = q.transpose(0, 1)
            cq = c.unsqueeze(2).expand(batch_size, c_len, q_len, -1) * \
                 q.unsqueeze(1).expand(batch_size, c_len, q_len, -1)
            cq = self.att_cq(cq).squeeze(3)  # (batch_size, c_len, q_len)

            s = self.att_c(c).expand(batch_size, c_len, q_len) + \
                self.att_q(q).expand(batch_size, q_len, c_len).transpose(1, 2) + \
                cq

            # 除掉空位
            mask = c_mask.eq(0)
            mask = mask.unsqueeze(2).expand(batch_size, c_len, q_len)
            s.masked_fill_(mask, -1e30)  # 使用-float('inf')过滤，会和dropout冲突，有nan值。 使用小值过滤，不同batch_size输出结果不同

            mask = q_mask.eq(0)
            mask = mask.unsqueeze(1).expand(batch_size, c_len, q_len)
            s.masked_fill_(mask, -1e30)

            # c2q
            a = f.softmax(s, dim=2)
            c2q = torch.bmm(a, q)  # (batch_size, c_len, hidden_size*2)

            # q2c
            b = torch.max(s, dim=2)[0]
            b = f.softmax(b, dim=1)  # (batch_size, c_len)
            q2c = torch.bmm(b.unsqueeze(1), c).expand(batch_size, c_len, -1)  # (batch_size, c_len, hidden_size*2)

            x = torch.cat([c, c2q, c * c2q, c * q2c], dim=2)
            x = c_mask.unsqueeze(2) * x
            x = x.transpose(0, 1)

            return x

        passage = batch[0: 3]
        query = batch[3: 6]
        alter_1 = batch[6]  # (batch_size, a1_len)
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
        alter1_vec = self.embedding(alter_1)  # (a1_len, batch_size, emb_size)
        alter2_vec = self.embedding(alter_2)
        alter3_vec = self.embedding(alter_3)

        # encoder: a_1
        alter1_vec = self.encoder_a(alter1_vec, alter1_mask)  # (a1_len, batch_size, hidden_size)
        alter1_alpha = self.a_att(alter1_vec).transpose(0, 1)  # (batch_size, a1_len, 1)
        mask_tmp = alter1_mask.eq(0).unsqueeze(2)
        alter1_alpha.masked_fill_(mask_tmp, -float('inf'))
        alter1_alpha = f.softmax(alter1_alpha, dim=1).transpose(1, 2)  # (batch_size, 1, a1_len)
        alter1_vec = torch.bmm(alter1_alpha, alter1_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size)

        # encoder: a_2
        alter2_vec = self.encoder_a(alter2_vec, alter2_mask)
        alter2_alpha = self.a_att(alter2_vec).transpose(0, 1)
        mask_tmp = alter2_mask.eq(0).unsqueeze(2)
        alter2_alpha.masked_fill_(mask_tmp, -float('inf'))
        alter2_alpha = f.softmax(alter2_alpha, dim=1).transpose(1, 2)
        alter2_vec = torch.bmm(alter2_alpha, alter2_vec.transpose(0, 1)).squeeze(1)

        # encoder: a_3
        alter3_vec = self.encoder_a(alter3_vec, alter3_mask)
        alter3_alpha = self.a_att(alter3_vec).transpose(0, 1)
        mask_tmp = alter3_mask.eq(0).unsqueeze(2)
        alter3_alpha.masked_fill_(mask_tmp, -float('inf'))
        alter3_alpha = f.softmax(alter3_alpha, dim=1).transpose(1, 2)
        alter3_vec = torch.bmm(alter3_alpha, alter3_vec.transpose(0, 1)).squeeze(1)

        # encoder: p
        passage_vec = self.encoder_p(passage_vec, passage_mask)  # (p_len, batch_size, hidden_size*2)
        passage_vec = self.dropout(passage_vec)

        # encoder: q
        query_vec = self.encoder_q(query_vec, query_mask)
        query_vec = self.dropout(query_vec)

        # attention: merge q into p
        g = att_flow_layer(passage_vec, passage_mask, query_vec, query_mask)  # (p_len, batch_size, hidden_size*8)
        m = self.modeling_rnn(g, passage_mask)  # (p_len, batch_size, hidden_size*2)
        gm = torch.cat([g, m], dim=2)  # (p_len, batch_size, hidden_size*10)

        # 获取q的自注意力表示
        sj = self.vq(torch.tanh(self.wq(query_vec))).transpose(0, 1)  # (batch_size, q_len, 1)
        mask_tmp = query_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, q_len)
        rq = torch.bmm(sj, query_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*2)
        rq = rq.unsqueeze(0)  # (1, batch_size, hidden_size*2)

        # 获取融合后p表示
        sj = self.vp(torch.tanh(self.wp1(gm) + self.wp2(rq))).transpose(0, 1)  # (batch_size, p_len, 1)
        mask_tmp = passage_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, p_len)
        rp = torch.bmm(sj, gm.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*10)

        # match
        rp = self.predict(rp)  # (batch_size, hidden_size)
        rp = f.leaky_relu(rp)
        rp = self.dropout(rp)
        rp = rp.unsqueeze(2)  # (batch_size, hidden_size, 1)
        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec]).transpose(0, 1)  # (batch_size, 3, hidden_size)
        outputs = torch.bmm(alters, rp).squeeze(2)  # (batch_size, 3)
        outputs = f.softmax(outputs, dim=1)

        return outputs
