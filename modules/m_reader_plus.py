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
    """ Reinforced Mnemonic Reader for Machine Reading Comprehension 2018 """
    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

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
        input_size = self.hidden_size * 2
        self.align_1 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, False)
        self.align_2 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, False)
        self.align_3 = Aligner(input_size, self.dropout_p, self.mode, self.is_bn, True)

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

        # attention: merge q into p
        R1, Z1, E1, B1 = self.align_1(passage_vec, passage_mask, query_vec, query_mask)
        R2, Z2, E2, B2 = self.align_2(R1, passage_mask, query_vec, query_mask, E1, B1)
        R3, _, _, _ = self.align_3(R2, passage_mask, query_vec, query_mask, E2, B2, Z1, Z2)
        new_p_vec = R3

        # 获取q的自注意力表示
        sj = self.vq(torch.tanh(self.wq(query_vec))).transpose(0, 1)  # (batch_size, q_len, 1)
        mask_tmp = query_mask.eq(0).unsqueeze(2)
        sj.masked_fill_(mask_tmp, -float('inf'))
        sj = f.softmax(sj, dim=1).transpose(1, 2)  # (batch_size, 1, q_len)
        rq = torch.bmm(sj, query_vec.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size*2)
        rq = rq.unsqueeze(0)  # (1, batch_size, hidden_size*2)

        # 获取融合后p表示
        sj = self.vp(torch.tanh(self.wp1(new_p_vec) + self.wp2(rq))).transpose(0, 1)  # (batch_size, p_len, 1)
        mask_tmp = passage_mask.eq(0).unsqueeze(2)
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


class Aligner(nn.Module):
    def __init__(self, input_size, dropout_p, mode, is_bn, use_rnn):
        super(Aligner, self).__init__()

        self.inter_align = InterAlign(input_size, dropout_p)
        self.self_align = SelfAlign(input_size, dropout_p)
        self.aggregation = EviCollection(mode, input_size, dropout_p, is_bn)

        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = encoder.Rnn(
                mode=mode,
                input_size=input_size*3,
                hidden_size=input_size//2,
                dropout_p=dropout_p,
                bidirectional=True,
                layer_num=1,
                is_bn=is_bn
            )

    def forward(self, U, U_mask, V, V_mask, E=None, B=None, Z1=None, Z2=None):
        """
        :param U: (c_len, batch_size, input_size)
        :param V: (q_len, batch_size, input_size)
        :param E: (batch_size, c_len, q_len)
        :param B:
        :param Z1: (c_len, batch_size, input_size)
        :param Z2:
        :return:R,Z,E,B
        """
        h, E = self.inter_align(U, U_mask, V, V_mask, E, B)
        Z, B = self.self_align(h, U_mask, B)
        if self.use_rnn:
            z = torch.cat([Z1, Z2, Z], dim=2)
            R = self.rnn(z, U_mask)
        else:
            R = self.aggregation(Z, U_mask)

        return R, Z, E, B


class InterAlign(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(InterAlign, self).__init__()

        self.Wu = nn.Linear(input_size, input_size)
        self.Wv = nn.Linear(input_size, input_size)
        self.gamma = nn.Parameter(torch.tensor(3.0))
        self.sfu = SFU(input_size, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, content_vec, content_mask, question_vec, question_mask, E_t=None, B_t=None):
        """
        :param content_vec: (c_len, batch_size, input_size)
        :param content_mask: (batch_size, c_len)
        :param question_vec: (q_len, batch_size, input_size)
        :param question_mask: (batch_size, q_len)
        :param E_t: (batch_size, c_len, q_len)  or None
        :param B_t: (batch_size, c_len, c_len) or None
        :return: h(c_len, batch_size, input_size), E_t(batch_size, c_len, q_len)
        """

        content_vec = self.dropout(content_vec)
        question_vec = self.dropout(question_vec)

        # E_t
        content_vec_tmp = f.relu(self.Wu(content_vec)).transpose(0, 1)  # (batch_size, c_len, input_size)
        question_vec_tmp = f.relu(self.Wv(question_vec)).transpose(0, 1)  # (batch_size, q_len, input_size)
        E_0 = torch.bmm(content_vec_tmp, question_vec_tmp.transpose(1, 2))  # (batch_size, c_len, q_len)

        if E_t is not None:
            E_t_mask = content_mask.eq(0).unsqueeze(2).expand(E_t.size())
            E_t = E_t.masked_fill(E_t_mask, -float('inf'))
            E_t = f.softmax(E_t, dim=1)  # (batch_size, c_len, q_len)

            B_t_mask = content_mask.eq(0).unsqueeze(1).expand(B_t.size())
            B_t = B_t.masked_fill(B_t_mask, -float('inf'))
            B_t = f.softmax(B_t, dim=2)  # (batch_size, c_len, c_len)

            E_1 = torch.bmm(B_t, E_t)
            E_t = E_0 + self.gamma * E_1  # (batch_size, c_len, q_len)
        else:
            E_t = E_0

        # V_bar
        mask = question_mask.eq(0).unsqueeze(1).expand(E_t.size())
        E_tt = E_t.masked_fill(mask, -float('inf'))
        E_tt = f.softmax(E_tt, dim=2)  # (batch_size, c_len, q_len)
        question_vec_tmp = torch.bmm(E_tt, question_vec.transpose(0, 1))  # (batch_size, c_len, input_size)
        question_vec_tmp = question_vec_tmp.transpose(0, 1)  # (c_len, batch_size, input_size)

        # fusion
        h = self.sfu(content_vec, question_vec_tmp)  # (c_len, batch_size, input_size)

        return h, E_t


class SelfAlign(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(SelfAlign, self).__init__()

        self.W1 = nn.Linear(input_size, input_size)
        self.W2 = nn.Linear(input_size, input_size)
        self.gamma = nn.Parameter(torch.tensor(3.0))
        self.sfu = SFU(input_size, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, h, h_mask, B_t=None):
        """
        :param h: (c_len, batch_size, input_size)
        :param h_mask: (batch_size, c_len)
        :param B_t: (batch_size, c_len_1, c_len_2) or None
        :return: h(c_len, batch_size, input_size), B_t(batch_size, c_len, c_len)
        """

        h = self.dropout(h)  # (c_len_2, batch_size, input_size)

        # B_t
        h_tmp_1 = f.relu(self.W1(h)).transpose(0, 1)  # (batch_size, c_len_1, input_size)
        h_tmp_2 = f.relu(self.W2(h)).transpose(0, 1)  # (batch_size, c_len_2, input_size)
        B_0 = torch.bmm(h_tmp_1, h_tmp_2.transpose(1, 2))  # (batch_size, c_len_1, c_len_2)

        if B_t is not None:
            mask_1 = h_mask.eq(0).unsqueeze(2).expand(B_t.size())
            B_t_1 = B_t.masked_fill(mask_1, -float('inf'))
            B_t_1 = f.softmax(B_t_1, dim=1)  # (batch_size, c_len_1, c_len_2)

            mask_2 = h_mask.eq(0).unsqueeze(1).expand(B_t.size())
            B_t_2 = B_t.masked_fill(mask_2, -float('inf'))
            B_t_2 = f.softmax(B_t_2, dim=2)  # (batch_size, c_len_1, c_len_1)

            B_1 = torch.bmm(B_t_2, B_t_1)  # (batch_size, c_len_1, c_len_2)
            B_t = B_0 + self.gamma * B_1

        else:
            B_t = B_0

        # make dialog
        mask = torch.eye(h.size(0), dtype=torch.uint8).cuda()
        mask = mask.unsqueeze(0)
        B_t.masked_fill_(mask, 0.0)

        # h_bar
        mask = h_mask.eq(0).unsqueeze(1).expand(B_t.size())
        B_tt = B_t.masked_fill(mask, -float('inf'))
        B_tt = f.softmax(B_tt, dim=2)
        h_tmp = torch.bmm(B_tt, h.transpose(0, 1))  # (batch_size, c_len_1, input_size)
        h_tmp = h_tmp.transpose(0, 1)  # (c_len_1, batch_size, input_size)

        # fusion
        h = self.sfu(h, h_tmp)  # (c_len_1, batch_size, input_size)

        return h, B_t


class EviCollection(nn.Module):
    def __init__(self, mode, input_size, dropout_p, is_bn):
        super(EviCollection, self).__init__()

        self.rnn = encoder.Rnn(
            mode=mode,
            input_size=input_size,
            hidden_size=input_size//2,
            dropout_p=dropout_p,
            bidirectional=True,
            layer_num=1,
            is_bn=is_bn
        )

    def forward(self, z, z_mask):
        """
        :param z: (c_len, batch_size, input_size)
        :return: (c_len, batch_size, input_size)
        """

        o = self.rnn(z, z_mask)
        return o


class SFU(nn.Module):
    def __init__(self, input_size, dropout_p):
        super(SFU, self).__init__()
        self.Wr = nn.Linear(input_size*4, input_size, bias=False)
        self.Wg = nn.Linear(input_size*4, input_size, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, fusion):
        """
        :param inputs: (.., input_size)
        :param fusion: (.., fusion_size)
        :return: (.., input_size)
        """
        m = torch.cat([inputs, fusion, inputs*fusion, inputs-fusion], dim=-1)
        m = self.dropout(m)
        x = f.relu(self.Wr(m))
        g = torch.sigmoid(self.Wg(m))
        o = x*g + (1-g)*inputs

        return o
