# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import pointer
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

        # elmo 融合比例控制
        self.s1 = nn.Parameter(torch.Tensor([1, 1, 1]))
        self.r1 = nn.Parameter(torch.Tensor([0.3]))

        # encoder
        input_size = self.embedding.embedding_dim + 1024
        self.encoder_pq = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=False
        )

        input_size = input_size - 9
        self.encoder_a = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=1,
            is_bn=False
        )

        self.mean_q = pointer.AttentionPooling(self.hidden_size*2, self.hidden_size)
        self.mean_a = pointer.AttentionPooling(self.hidden_size*2, self.hidden_size)

        # merge q into p
        self.aligner = nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                          for _ in range(self.num_align_hops)])
        # self align
        self.self_aligner = nn.ModuleList([SelfSeqAtten() for _ in range(self.num_align_hops)])
        self.self_aligner_sfu = nn.ModuleList([SFU(self.hidden_size*2, self.hidden_size*2*3, dropout_p=self.dropout_p)
                                               for _ in range(self.num_align_hops)])
        # aggregation
        self.choose_agg = nn.ModuleList([encoder.Rnn(
            mode=self.mode,
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )
            for _ in range(self.num_align_hops)])

        # p_rep, choosing
        self.wp1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.wp2 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.vp = nn.Linear(self.hidden_size, 1)
        self.bi_linear = nn.Linear(self.hidden_size*2, self.hidden_size*2)

        self.dropout = nn.Dropout(self.dropout_p)

        self.reset_param()

    def reset_param(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, batch, elmo):
        """
        :param batch: (p_index, q_index, zhengli_index, fuli_index, wfqd_index)
        :return:
        """

        passage = batch[0: 3]
        query = batch[3: 6]
        zhengli = batch[6]  # (batch_size, zhengli_len)
        fuli = batch[7]
        wfqd = batch[8]

        # mask
        passage_mask = utils.get_mask(passage[0])
        query_mask = utils.get_mask(query[0])
        zhengli_mask = utils.get_mask(zhengli)
        fuli_mask = utils.get_mask(fuli)
        wfqd_mask = utils.get_mask(wfqd)

        # embedding
        passage_vec = self.embedding(passage)
        query_vec = self.embedding(query)
        zhengli_vec = self.embedding(zhengli)
        fuli_vec = self.embedding(fuli)
        wfqd_vec = self.embedding(wfqd)

        # elmo
        p_elmo, q_elmo, zhengli_elmo, fuli_elmo, wfqd_elmo = elmo
        s1 = f.softmax(self.s1, dim=0)

        p_elmo = self.r1*(s1[0]*p_elmo[0] + s1[1]*p_elmo[1] + s1[2]*p_elmo[2]).transpose(0, 1)  # (c_len, batch_size, 1024)
        passage_vec = torch.cat([passage_vec, p_elmo], dim=2)  # (c_len, batch_size, w2v+1024)

        q_elmo = self.r1*(s1[0]*q_elmo[0] + s1[1]*q_elmo[1] + s1[2]*q_elmo[2]).transpose(0, 1)
        query_vec = torch.cat([query_vec, q_elmo], dim=2)

        zhengli_elmo = self.r1*(s1[0]*zhengli_elmo[0] + s1[1]*zhengli_elmo[1] + s1[2]*zhengli_elmo[2]).transpose(0, 1)
        zhengli_vec = torch.cat([zhengli_vec, zhengli_elmo], dim=2)

        fuli_elmo = self.r1*(s1[0]*fuli_elmo[0] + s1[1]*fuli_elmo[1] + s1[2]*fuli_elmo[2]).transpose(0, 1)
        fuli_vec = torch.cat([fuli_vec, fuli_elmo], dim=2)

        wfqd_elmo = self.r1*(s1[0]*wfqd_elmo[0] + s1[1]*wfqd_elmo[1] + s1[2]*wfqd_elmo[2]).transpose(0, 1)
        wfqd_vec = torch.cat([wfqd_vec, wfqd_elmo], dim=2)

        # encoder
        passage_vec = self.encoder_pq(passage_vec, passage_mask)  # (p_len, batch_size. hidden_size*2)
        passage_vec = self.dropout(passage_vec)
        query_vec = self.encoder_pq(query_vec, query_mask)
        query_vec = self.dropout(query_vec)
        zhengli_vec = self.encoder_a(zhengli_vec, zhengli_mask)
        zhengli_vec = self.dropout(zhengli_vec)
        fuli_vec = self.encoder_a(fuli_vec, fuli_mask)
        fuli_vec = self.dropout(fuli_vec)
        wfqd_vec = self.encoder_a(wfqd_vec, wfqd_mask)
        wfqd_vec = self.dropout(wfqd_vec)

        # answer build
        zhengli_vec = self.mean_a(zhengli_vec, zhengli_mask)  # (batch_size, hidden_size*2)
        fuli_vec = self.mean_a(fuli_vec, fuli_mask)
        wfqd_vec = self.mean_a(wfqd_vec, wfqd_mask)
        answer = torch.stack([zhengli_vec, fuli_vec, wfqd_vec]).transpose(0, 1)  # (batch_size, 3, hidden_size*2)

        # merge q into p, get p prep
        align_ct = passage_vec
        for i in range(self.num_align_hops):
            qt_align_ct = self.aligner[i](align_ct, query_vec, query_mask)
            bar_ct = self.aligner_sfu[i](align_ct,
                                         torch.cat([qt_align_ct, align_ct*qt_align_ct, align_ct-qt_align_ct], dim=2))

            ct_align_ct = self.self_aligner[i](bar_ct, passage_mask)
            hat_ct = self.self_aligner_sfu[i](bar_ct,
                                              torch.cat([ct_align_ct, bar_ct*ct_align_ct, bar_ct-ct_align_ct], dim=2))
            align_ct = self.choose_agg[i](hat_ct, passage_mask)
        p_prep = align_ct  # (p_len, batch_size, hidden_size*2)
        q_prep = self.mean_q(query_vec, query_mask).unsqueeze(0)  # (1, batch_size, hidden_size*2)
        sj = self.vp(torch.tanh(self.wp1(p_prep) + self.wp2(q_prep))).transpose(0, 1)  # (batch_size, p_len, 1)
        mask = passage_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, p_len)
        p_prep = torch.bmm(sj, p_prep.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*2)

        # choosing
        p_prep = self.bi_linear(p_prep)  # (batch_size, hidden_size*2)
        outputs = torch.bmm(answer, p_prep.unsqueeze(2)).squeeze(2)  # (batch_size, 3)

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
