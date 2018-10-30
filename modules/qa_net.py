# encoding = utf-8
# author = xy

import math
import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
import utils

Max_Content_len = 500
Max_Question_len = 150


class Model(nn.Module):
    """ qa-net for reading comprehension"""
    def __init__(self, param):
        super(Model, self).__init__()

        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.w2i = param['embedding'].shape[1]
        self.hidden_size = param['hidden_size']

        self.embedding = embedding.ExtendEmbedding(param['embedding'])
        self.flag = True
        if self.flag:
            self.highway_c = Highway(self.w2i+6)
            self.highway_q = Highway(self.w2i+4)

        self.content_conv = DepthwiseSeparableConv(self.w2i+6, self.hidden_size, 5)
        self.question_conv = DepthwiseSeparableConv(self.w2i+4, self.hidden_size, 5)

        self.c_enc = EncoderBlock(conv_num=2, d=self.hidden_size, k=5, length=Max_Content_len, dropout_p=self.dropout_p)
        self.q_enc = EncoderBlock(conv_num=2, d=self.hidden_size, k=5, length=Max_Question_len, dropout_p=self.dropout_p)

        self.cq_att = CQAttention(self.hidden_size, self.dropout_p)

        self.cq_resizer = DepthwiseSeparableConv(self.hidden_size*4, self.hidden_size, 5)

        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, d=self.hidden_size, k=5, length=Max_Content_len,
                                            dropout_p=self.dropout_p) for _ in range(3)])

        self.pointer = Pointer(self.hidden_size)

    def forward(self, batch):
        """
        :param batch:
        :return: (2, batch_size, c_len)
        """
        content = batch[0: 4]
        question = batch[4: 6]

        # mask
        content_mask = utils.get_mask(content[0])
        question_mask = utils.get_mask(question[0])

        # embedding
        content = self.embedding(content, True)  # (c_len, batch_size, w2i_size+6)
        question = self.embedding(question, False)  # (q_len, batch_size, w2i_size+4)

        # embedding done
        if self.flag:
            content = f.dropout(content, p=self.encoder_dropout_p, training=self.training)
            question = f.dropout(question, p=self.encoder_dropout_p, training=self.training)
            content = self.highway_c(content)
            question = self.highway_q(question)  # (q_len, batch_size, w2i_size+4)

        # conv
        content = content.transpose(0, 1).transpose(1, 2)  # (batch_size, w2i_size+6, c_len)
        question = question.transpose(0, 1).transpose(1, 2)  # (batch_size, w2i_size+4, q_len)
        content = self.content_conv(content)  # (batch_size, hidden_size, c_len)
        question = self.question_conv(question)  # (batch_size, hidden_size, q_len)

        # encoder
        content = self.c_enc(content, content_mask)
        question = self.q_enc(question, question_mask)

        # cq attention
        X = self.cq_att(content, content_mask, question, question_mask)  # (batch_size, d*4, c_len)

        # model encoder layer
        M0 = self.cq_resizer(X)  # (batch_size, d, c_len)
        for enc in self.model_enc_blks:
            M0 = enc(M0, content_mask)
        M1 = M0
        for enc in self.model_enc_blks:
            M1 = enc(M1, content_mask)
        M2 = M1
        for enc in self.model_enc_blks:
            M2 = enc(M2, content_mask)

        result = self.pointer(M0, M1, M2, content_mask)

        return result


class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()

        self.size = size
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(2)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(2)])

    def forward(self, x):
        """
        :param x: (*, *, w2i_size)
        :return: (*, *, w2i_size)
        """
        for i in range(2):
            gate = f.sigmoid(self.gate[i](x))
            nonlinear = f.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        """
        :param x:  (batch_size, w2i, c_len)
        :return: (batch_size, d, c_len)
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, d, heads_num):
        super(SelfAttention, self).__init__()

        self.heads_num = heads_num
        self.dv = d // heads_num

        self.W0 = nn.Linear(d, d)
        self.Wqs = nn.ModuleList([torch.nn.Linear(d, self.dv) for _ in range(heads_num)])
        self.Wks = nn.ModuleList([torch.nn.Linear(d, self.dv) for _ in range(heads_num)])
        self.Wvs = nn.ModuleList([torch.nn.Linear(d, self.dv) for _ in range(heads_num)])

    def forward(self, x, mask):
        """
        :param x: (batch_size, d, c_len)
        :param mask: (batch_size, c_len)
        :return: (batch_size, d, c_len)
        """
        x = x.transpose(1, 2)  # (batch_size, c_len, d)
        h_mask = mask.unsqueeze(1)
        v_mask = mask.unsqueeze(2)
        heads = []
        for i in range(self.heads_num):
            wqs_i = self.Wqs[i](x)  # (batch_size, c_len, dv)
            wks_i = self.Wks[i](x)
            wvs_i = self.Wvs[i](x)
            out = torch.bmm(wqs_i, wks_i.transpose(1, 2))  # (batch_size, c_len, c_len)
            out = out * (1 / math.sqrt(self.dv))
            out = utils.mask_logits(out, h_mask)
            out = f.softmax(out, dim=2)
            out = out * v_mask
            out = torch.bmm(out, wvs_i)  # (batch_size, c_len, dv)
            heads.append(out)
        heads = torch.cat(heads, dim=2)  # (batch_size, c_len, d)
        heads = self.W0(heads).transpose(1, 2)  # (batch_size, d, c_len)
        return heads


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d, k, length, dropout_p):
        super(EncoderBlock, self).__init__()
        self.conv_num = conv_num
        self.d = d
        self.k = k
        self.length = length
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList([DepthwiseSeparableConv(d, d, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d, 8)
        self.fc = nn.Linear(d, d)
        self.pos = PosEncoder(length, d)
        self.norm_1 = nn.LayerNorm(d)
        self.norm_2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(conv_num)])
        self.norm_3 = nn.LayerNorm(d)
        self.L = conv_num

    def forward(self, x, mask):
        """
        :param x: (batch_size, d, len)
        :param mask: (batch_size, len)
        :return: (batch_size, d, len)
        """
        out = self.pos(x)
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = f.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout_p * (i + 1) / self.L
                out = f.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norm_2[i](out.transpose(1, 2)).transpose(1, 2)
        out = self.self_att(out, mask)
        out = out + res
        out = f.dropout(out, p=self.dropout_p, training=self.training)
        res = out
        out = self.norm_3(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = f.relu(out)
        out = out + res
        out = f.dropout(out, p=self.dropout_p, training=self.training)

        return out


class PosEncoder(nn.Module):
    def __init__(self, length, hidden_size):
        super(PosEncoder, self).__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / hidden_size) if i % 2 == 0 else -10000 ** ((1 - i) / hidden_size) for i in range(hidden_size)])
        freqs = freqs.unsqueeze(1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(hidden_size)])
        phases = phases.unsqueeze(1)
        pos = torch.arange(length).repeat(hidden_size, 1)
        self.pos_encoding = nn.Parameter(torch.sin(pos*freqs + phases), requires_grad=False).cuda()

    def forward(self, x):
        """
        :param x: (batch_size, d, c_len)
        :return: (batch_size, d, c_len)
        """
        c_len = x.size(2)
        pos_enc = self.pos_encoding[:, :c_len]
        x = x + pos_enc
        return x


class CQAttention(nn.Module):
    def __init__(self, d, dropout_p):
        super(CQAttention, self).__init__()
        self.d = d
        self.dropout_p = dropout_p
        self. w = nn.Linear(d*3, 1)

    def forward(self, c, c_mask, q, q_mask):
        """
        :param c: (batch_size, d, c_len)
        :param c_mask: (batch_size, c_len)
        :param q: (batch_size, d, q_len)
        :param q_mask: (batch_size, q_len)
        :return: (batch_size, d*4, c_len)
        """
        batch_size = c.size(0)
        c_len = c.size(2)
        q_len = q.size(2)

        c = c.transpose(1, 2)  # (batch_size, c_len, d)
        q = q.transpose(1, 2)  # (batch_size, q_len, d)

        c_e = c.unsqueeze(2).expand(batch_size, c_len, q_len, self.d)
        q_e = q.unsqueeze(1).expand(batch_size, c_len, q_len, self.d)
        cq = c_e * q_e

        s = torch.cat([q_e, c_e, cq], dim=3)
        s = self.w(s).squeeze()  # (batch_size, c_len, q_len)

        c_mask1 = c_mask.unsqueeze(2).expand(batch_size, c_len, q_len)
        s1 = utils.mask_logits(s, c_mask1)
        s1 = f.softmax(s1, dim=2)

        q_mask = q_mask.unsqueeze(1).expand(batch_size, c_len, q_len)
        s2 = utils.mask_logits(s, q_mask)
        s2 = f.softmax(s2, dim=1)

        A = torch.bmm(s1, q)  # (batch_size, c_len, d)
        B = torch.bmm(s1, s2.transpose(1, 2))  # (batch_size, c_len, c_len)
        B = torch.bmm(B, c)  # (batch_size, c_len, d)

        out = torch.cat([c, A, c*A, c*B], dim=2)  # (batch_size, c_len, d*4)
        out = f.dropout(out, p=self.dropout_p, training=self.training)
        out = out.transpose(1, 2)  # (batch_size, d*3, c_len)

        return out


class Pointer(nn.Module):
    def __init__(self, d):
        super(Pointer, self).__init__()
        self.d = d

        self.w1 = nn.Linear(d*2, 1)
        self.w2 = nn.Linear(d*2, 1)

    def forward(self, M0, M1, M2, mask):
        """
        :param M0: (batch_size, d, c_len)
        :param M1:
        :param M2:
        :param mask: (batch_size, c_len)
        :return: (2, batch_size, c_len)
        """

        M0 = M0.transpose(1, 2)  # (batch_size, c_len, d)
        M1 = M1.transpose(1, 2)
        M2 = M2.transpose(1, 2)

        x1 = torch.cat([M0, M1], dim=2)
        x2 = torch.cat([M0, M2], dim=2)

        mask_1 = mask.eq(0)
        x1 = self.w1(x1).squeeze()  # (batch_size, c_len)
        x1.masked_fill_(mask_1, -float('inf'))
        x1 = f.softmax(x1, dim=1)

        x2 = self.w2(x2).squeeze()
        x2.masked_fill_(mask_1, -float('inf'))
        x2 = f.softmax(x2, dim=1)

        result = torch.stack([x1, x2])

        result = result + torch.ones(result.size()).cuda() * 1e-30

        return result
