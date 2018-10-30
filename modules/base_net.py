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
    """ AIC 官网提供的 baseline"""

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

        # Concat Attention
        self.Wc1 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.Wc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vc = nn.Linear(self.hidden_size, 1, bias=False)

        # Bilinear Attention
        self.Wb = nn.Linear(self.hidden_size*2, self.hidden_size*2, bias=False)

        # Dot Attention :
        self.Wd = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vd = nn.Linear(self.hidden_size, 1, bias=False)

        # Minus Attention :
        self.Wm = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vm = nn.Linear(self.hidden_size, 1, bias=False)

        self.gru_agg = encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size*10,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=self.is_bn
        )

        # prediction
        self.wq = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vq = nn.Linear(self.hidden_size, 1, bias=False)
        self.wp1 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.wp2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.vp = nn.Linear(self.hidden_size, 1, bias=False)
        self.predict = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

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
        passage_vec = self.dropout(passage_vec).transpose(0, 1)  # (batch_size, p_len, hidden_size*2)

        # encoder: q
        query_vec = self.encoder_q(query_vec, query_mask)
        query_vec = self.dropout(query_vec).transpose(0, 1)  # （batch_size, q_len, hidden_size*2）

        # attention: cat
        p1 = self.Wc1(passage_vec).unsqueeze(2)  # (batch_size, p_len, 1, hidden_size)
        q1 = self.Wc2(query_vec).unsqueeze(1)  # (batch_size, 1, q_len, hidden_size)
        alpha = self.vc(torch.tanh(p1 + q1)).squeeze(3)  # (batch_size, p_len, q_len)
        mask_tmp = query_mask.eq(0).unsqueeze(1).expand(alpha.size())  # (batch_size, p_len, q_len)
        alpha.masked_fill_(mask_tmp, -float('inf'))
        alpha = f.softmax(alpha, dim=2)  # (batch_size, p_len, q_len)
        pc = torch.bmm(alpha, query_vec)  # (batch_size, p_len, hidden_size*2)

        # attention: bilinear
        p2 = self.Wb(passage_vec)  # (batch_size, p_len, hidden_size*2)
        alpha = torch.bmm(p2, query_vec.transpose(1, 2))  # (batch_size, p_len, q_len)
        alpha.masked_fill_(mask_tmp, -float('inf'))
        alpha = f.softmax(alpha, dim=2)  # (batch_size, p_len, q_len)
        pb = torch.bmm(alpha, query_vec)  # (batch_size, p_len, hidden_size*2)

        # attention: dot
        p3 = passage_vec.unsqueeze(2)  # (batch_size, p_len, 1, hidden_size*2)
        q3 = query_vec.unsqueeze(1)  # (batch_size, 1, q_len, hidden_size*2)
        alpha = self.vd(torch.tanh(self.Wd(p3 * q3))).squeeze(3)  # (batch_size, p_len, q_len)
        alpha.masked_fill_(mask_tmp, -float('inf'))
        alpha = f.softmax(alpha, dim=2)  # (batch_size, p_len, q_len)
        pd = torch.bmm(alpha, query_vec)  # (batch_size, p_len, hidden_size*2)

        # attention: minus
        p4 = passage_vec.unsqueeze(2)
        q4 = query_vec.unsqueeze(1)
        alpha = self.vm(torch.tanh(self.Wm(q4-p4))).squeeze(3)
        alpha.masked_fill_(mask_tmp, -float('inf'))
        alpha = f.softmax(alpha, dim=2)
        pm = torch.bmm(alpha, query_vec)

        # attention: self
        # p5 = passage_vec.unsqueeze(2)
        # q5 = passage_vec.unsqueeze(1)
        # alpha = self.vs(torch.tanh(self.Ws(p5 * q5))).squeeze(3)
        # alpha.masked_fill_(mask_tmp, -float('inf'))
        # alpha = f.softmax(alpha, dim=2)
        # ps = torch.bmm(alpha, passage_vec)

        # aggregation
        aggregation = torch.cat([passage_vec, pc, pb, pd, pm], dim=2)
        aggregation = self.gru_agg(aggregation.transpose(0, 1), passage_mask).transpose(0, 1)  # (batch_size, p_len, hidden_size)

        # 获取q的自注意力表示
        sj = self.vq(torch.tanh(self.wq(query_vec)))  # (batch_size, q_len, 1)
        mask_tmp = query_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, q_len)
        rq = torch.bmm(sj, query_vec).squeeze(1)  # (batch_size, hidden_size*2)
        rq = rq.unsqueeze(1)  # (batch_size, 1, hidden_size*2)

        # 获取融合后p表示
        sj = self.vp(torch.tanh(self.wp1(aggregation) + self.wp2(rq)))  # (batch_size, p_len, 1)
        mask_tmp = passage_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, p_len)
        rp = torch.bmm(sj, aggregation).squeeze(1)  # (batch_size, hidden_size)

        # match
        rp = self.predict(rp)  # (batch_size, hidden_size)
        rp = f.leaky_relu(rp)
        rp = self.dropout(rp)
        rp = rp.unsqueeze(2)  # (batch_size, hidden_size, 1)
        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec]).transpose(0, 1)  # (batch_size, 3, hidden_size)
        outputs = torch.bmm(alters, rp).squeeze(2)  # (batch_size, 3)
        outputs = f.softmax(outputs, dim=1)

        return outputs
