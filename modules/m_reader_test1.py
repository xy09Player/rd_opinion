# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import pointer
from modules.layers import choose
import utils


class Model(nn.Module):
    """
    paper: Reinforced Mnemonic Reader for Machine Comprehension 2017
    """
    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']
        self.num_align_hops = param['num_align_hops']  # 2

        # embedding
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

        # align
        self.aligner = nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                          for _ in range(self.num_align_hops)])

        # self align
        self.self_aligner = nn.ModuleList([SelfSeqAtten() for _ in range(self.num_align_hops)])
        self.self_aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                               for _ in range(self.num_align_hops)])

        # aggregation
        self.aggregation = nn.ModuleList([encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )
            for _ in range(self.num_align_hops)])

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

        # attention: merge p into q
        align_ct = query_vec
        for i in range(self.num_align_hops):
            qt_align_ct = self.aligner[i](align_ct, passage_vec, passage_mask)
            bar_ct = self.aligner_sfu[i](align_ct,
                                         torch.cat([qt_align_ct, align_ct*qt_align_ct, align_ct-qt_align_ct], dim=2))

            ct_align_ct = self.self_aligner[i](bar_ct, query_mask)
            hat_ct = self.self_aligner_sfu[i](bar_ct,
                                              torch.cat([ct_align_ct, bar_ct*ct_align_ct, bar_ct-ct_align_ct], dim=2))
            align_ct = self.aggregation[i](hat_ct, query_mask)
        new_p_vec = align_ct  # (p_len, batch_size, hidden_size*2)

        # 获取q的自注意力表示
        sj = self.vq(torch.tanh(self.wq(query_vec))).transpose(0, 1)  # (batch_size, q_len, 1)
        mask_tmp = query_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, q_len)
        rq = torch.bmm(sj, query_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*2)
        rq = rq.unsqueeze(0)  # (1, batch_size, hidden_size*2)

        # 获取融合后p表示
        sj = self.vp(torch.tanh(self.wp1(new_p_vec) + self.wp2(rq))).transpose(0, 1)  # (batch_size, p_len, 1)
        mask_tmp = query_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, p_len)
        rp = torch.bmm(sj, new_p_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*10)

        # match
        rp = self.predict(rp)  # (batch_size, hidden_size)
        rp = f.leaky_relu(rp)
        rp = self.dropout(rp)
        rp = rp.unsqueeze(2)  # (batch_size, hidden_size, 1)
        alters = torch.stack([alter1_vec, alter2_vec, alter3_vec]).transpose(0, 1)  # (batch_size, 3, hidden_size)
        outputs = torch.bmm(alters, rp).squeeze(2)  # (batch_size, 3)
        outputs = f.softmax(outputs, dim=1)

        return outputs


class SeqToSeqAtten(nn.Module):
    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, content_vec, question_vec, question_mask):
        """
        :param content_vec: (c_len, batch_size, hidden_size)
        :param question_vec:
        :param question_mask:
        :return: (c_len, batch_size, hidden_size)
        """
        content_vec = content_vec.transpose(0, 1)  # (batch_size, c_len, hidden_size)
        question_vec = question_vec.transpose(0, 1)

        b = torch.bmm(content_vec, question_vec.transpose(1, 2))  # (batch_size, c_len, q_len)

        # mask
        mask = question_mask.eq(0).unsqueeze(1).expand(b.size())  # (batch_size, c_len, q_len)
        b.masked_fill_(mask, -float('inf'))

        b = f.softmax(b, dim=2)
        q = torch.bmm(b, question_vec)  # (batch_size, c_len, hidden_size)
        q = q.transpose(0, 1)  # (c_len, batch_size, hidden_size)

        return q


class SFU(nn.Module):
    def __init__(self, input_size, fusion_size, dropout_p):
        super(SFU, self).__init__()

        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, fusions):
        """
        :param inputs:  (c_len, batch_size, input_size)
        :param fusions: (c_len, batch_size, input_size*3)
        :return: (c_len, batch_size, input_size)
        """

        m = torch.cat([inputs, fusions], dim=-1)
        m = self.dropout(m)
        r = torch.tanh(self.linear_r(m))
        g = torch.sigmoid(self.linear_g(m))
        o = g * r + (1-g) * inputs

        return o


class SelfSeqAtten(nn.Module):
    def __init__(self):
        super(SelfSeqAtten, self).__init__()

    def forward(self, h, h_mask):
        """
        :param h: (c_len, batch_size, input_size)
        :param h_mask: (batch_size, c_len)
        :return: (c_len, batch_size, input_size)
        """
        c_len = h.size(0)

        h = h.transpose(0, 1)  # (batch_size, c_len, input_size)
        alpha = torch.bmm(h, h.transpose(1, 2))  # (batch_size, c_len, c_len)

        # mask dialog
        mask = torch.eye(c_len, dtype=torch.uint8).cuda()
        mask = mask.unsqueeze(0)
        alpha.masked_fill_(mask, 0.0)

        # mask inf
        mask = h_mask.eq(0).unsqueeze(1).expand(alpha.size())  # (batch_size, c_len, c_len)
        alpha.masked_fill_(mask, -float('inf'))

        alpha = f.softmax(alpha, dim=2)
        o = torch.bmm(alpha, h)  # (batch_size, c_len, input_size)
        o = o.transpose(0, 1)

        return o
