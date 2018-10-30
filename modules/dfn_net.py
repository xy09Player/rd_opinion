# coding = utf-8
# author = xy
"""
paper "Dynamic Fusion networks for machine reading comprehension"
简化版：
1. 去掉了多注意选择模式，仅使用第三种 Entangled attention
2. 去掉了迭代次数控制机制，调整为static， 迭代次数为超参数
"""

import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder
import utils


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']
        self.k = param['k']
        self.perspective_num = 10
        self.lamda = 10

        self.embedding = embedding.ExtendEmbedding(param['embedding'])

        # encoder: p, q, a
        input_size = self.embedding.embedding_dim - 9
        self.encoder = encoder.Rnn(
            mode='LSTM',
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=False,
        )

        # attention: q2p
        self.q2p = MultiPerspective(self.hidden_size, self.perspective_num, self.dropout_p)

        # attention: a2p
        self.a2p = MultiPerspective(self.hidden_size, self.perspective_num, self.dropout_p)

        # attention: Mq2Ma
        self.mq2ma = MultiPerspective(self.perspective_num*4, self.perspective_num, self.dropout_p)

        # attention: Ma2Mq
        self.ma2mq = MultiPerspective(self.perspective_num*4, self.perspective_num, self.dropout_p)

        # aggregation
        self.aggregation = Aggregation(self.perspective_num, self.hidden_size, self.dropout_p)

        # memory
        self.q2p = MultiPerspective(self.hidden_size, self.perspective_num, self.dropout_p)
        self.m_rnn = encoder.Rnn(
            mode='LSTM',
            input_size=self.perspective_num*8,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            layer_num=1,
            is_bn=self.is_bn,
        )

        # answer score
        self.answer_score = AnswerScore(self.hidden_size, self.k, self.lamda, self.dropout_p)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, batch):
        passage = batch[0]
        query = batch[3]
        alter_1 = batch[6]
        alter_2 = batch[7]
        alter_3 = batch[8]

        flag = False
        index = -1
        for i in range(alter_1.size(0)):
            if alter_1[i][0] == alter_2[i][0] == alter_3[i][0]:
                print('1', alter_1[i])
                print('2', alter_2[i])
                print('3', alter_3[i])
                index = i
                flag = True





        # mask
        passage_mask = utils.get_mask(passage)
        query_mask = utils.get_mask(query)
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
        passage_vec = self.encoder(passage_vec, passage_mask).transpose(0, 1)  # (batch_size, p_len, hidden_size*2)
        query_vec = self.encoder(query_vec, query_mask).transpose(0, 1)  # (batch_size, q_len, hidden_size*2)
        alter1_vec = self.encoder(alter1_vec, alter1_mask).transpose(0, 1)  # (batch_size, a1_len, hidden_size*2)
        alter2_vec = self.encoder(alter2_vec, alter2_mask).transpose(0, 1)
        alter3_vec = self.encoder(alter3_vec, alter3_mask).transpose(0, 1)

        # attention
        mq = self.q2p(query_vec, passage_vec, query_mask, passage_mask)  # (batch_size, q_len, perspective_num*8)
        ma1 = self.a2p(alter1_vec, passage_vec, alter1_mask, passage_mask)  # (batch_size, a1_len, perspective_num*8)
        ma2 = self.a2p(alter2_vec, passage_vec, alter2_mask, passage_mask)
        ma3 = self.a2p(alter3_vec, passage_vec, alter3_mask, passage_mask)

        q1 = self.mq2ma(mq, ma1, query_mask, alter1_mask)  # (batch_size, q_len, perspective_num*8)
        q2 = self.mq2ma(mq, ma2, query_mask, alter2_mask)
        q3 = self.mq2ma(mq, ma3, query_mask, alter3_mask)

        if flag:
            print('q1', q1[index][0][0].item())
            print('q2', q2[index][0][0].item())
            print('q3', q3[index][0][0].item())

        # q1 = self.mq2ma(mq, ma2, query_mask, alter2_mask)
        a1 = self.ma2mq(ma1, mq, alter1_mask, query_mask)  # (batch_size, a1_len, perspective_num*8)
        s1 = self.aggregation(q1, a1, query_mask, alter1_mask)  # (batch_size, hidden_size*4)

        if flag:
            print('alter1_mask', alter1_mask[index])




        # q2 = self.mq2ma(mq, ma2, query_mask, alter2_mask)
        # q2 = self.mq2ma(mq, ma1, query_mask, alter1_mask)

        a2 = self.ma2mq(ma2, mq, alter2_mask, query_mask)
        s2 = self.aggregation(q2, a2, query_mask, alter2_mask)

        if flag:
            print('alter2_mask', alter2_mask[index])


        # q3 = self.mq2ma(mq, ma3, query_mask, alter3_mask)
        # q3 = self.mq2ma(mq, ma1, query_mask, alter1_mask)
        a3 = self.ma2mq(ma3, mq, alter3_mask, query_mask)
        s3 = self.aggregation(q3, a3, query_mask, alter3_mask)

        if flag:
            print('alter3_mask', alter3_mask[index])

        if flag:
            print('q1', q1[index][0][0].item())
            print('q2', q2[index][0][0].item())
            print('q3', q3[index][0][0].item())


        if flag:

            print('s1', s1[index][0].item())
            print('s2', s2[index][0].item())
            print('s3', s3[index][0].item())

            print('s2==s3?', s2[index][0].item() == s3[index][0].item())


        # memory
        m = self.q2p(query_vec, passage_vec, query_mask, passage_mask)  # (batch_size, q_len, perspective_num*8)
        m = m.transpose(0, 1)
        m = self.m_rnn(m, query_mask).transpose(0, 1)  # (batch_size, q_len, hidden_size*2)

        # answer
        outputs = self.answer_score(m, s1, s2, s3, query_mask)  # (batch_size, 3)
        outputs = f.log_softmax(outputs, dim=1)


        # if flag:
        #     print(outputs[index])

        return outputs


class MultiPerspective(nn.Module):
    def __init__(self, hidden_size, perspective_num, dropout_p):
        super(MultiPerspective, self).__init__()

        self.hidden_size = hidden_size
        self.perspective_num = perspective_num

        self.fw_1 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.bw_1 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.fw_2 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.bw_2 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.fw_3 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.bw_3 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.fw_4 = nn.Parameter(torch.rand(perspective_num, hidden_size))
        self.bw_4 = nn.Parameter(torch.rand(perspective_num, hidden_size))

        self.dropout = nn.Dropout(dropout_p)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fw_1)
        nn.init.kaiming_normal_(self.bw_1)
        nn.init.kaiming_normal_(self.fw_2)
        nn.init.kaiming_normal_(self.bw_2)
        nn.init.kaiming_normal_(self.fw_3)
        nn.init.kaiming_normal_(self.bw_3)
        nn.init.kaiming_normal_(self.fw_4)
        nn.init.kaiming_normal_(self.bw_4)

    def _full_matching(self, v1, v2, w):
        """
        :param v1: (batch_size, v1_len, hidden_size)
        :param v2: (batch_size, v1_len, hidden_size) or (batch_size, hidden_size)
        :param w: (perspective_num, hidden_size)
        :return: (batch_size, v1_len, perspective_num)
        """
        batch_size = v1.size(0)
        v1_len = v1.size(1)
        hidden_size = v1.size(2)
        perspective_num = w.size(0)

        # (1, 1, hidden_size, perspective_num)
        w = w.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        # (batch_size, v1_len, hidden_size, perspective_num)
        v1 = v1.unsqueeze(3).expand(batch_size, v1_len, hidden_size, perspective_num) * w
        if len(v2.size()) == 3:
            v2 = v2.unsqueeze(3).expand(batch_size, v1_len, hidden_size, perspective_num) * w
        else:
            v2 = v2.unsqueeze(1).expand(batch_size, v1_len, hidden_size)
            v2 = v2.unsqueeze(3).expand(batch_size, v1_len, hidden_size, perspective_num) * w

        # (batch_size, v1_len, perspective_num)
        m = f.cosine_similarity(v1, v2, dim=2)

        return m

    def _maxpooling_matching(self, v1, v2, w):
        """
        实现方式跟paper中的不太一样
        :param v1: (batch_size, v1_len, hidden_size)
        :param v2: (batch_size, v2_len, hidden_size)
        :param w: (perspective_num, hidden_size)
        :return: (batch_size, v1_len, perspective_num)
        """
        batch_size = v1.size(0)
        v1_len = v1.size(1)
        v2_len = v2.size(1)
        hidden_size = v1.size(2)
        perspective_num = w.size(0)

        # (1, perspective_num, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)
        # (batch_size, perspective_num, seq_len, hidden_size)
        v1 = v1.unsqueeze(1).expand(batch_size, perspective_num, v1_len, hidden_size) * w
        v2 = v2.unsqueeze(1).expand(batch_size, perspective_num, v2_len, hidden_size) * w
        # (batch_size, perspective_num, seq_len, 1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)

        # (batch_size, perspective_num, v1_len, v2_len)
        n = torch.matmul(v1, v2.transpose(2, 3))
        d = v1_norm * v2_norm.transpose(2, 3)

        # (batch_size, v1_len, v2_len, perspective_num)
        m = self._div_with_small_value(n, d).permute(0, 2, 3, 1)
        m, _ = m.max(dim=2)

        return m

    def _attention_matching(self, v1, v2, w, w2):
        """
        实现方式跟paper中的不太一样
        :param v1: (batch_size, v1_len, hidden_size)
        :param v2: (batch_size, v2_len, hidden_size)
        :param w: (perspective_num, hidden_size)
        :return: (batch_size, v1_len, perspective_num) and (batch_size, v1_len, perspective_num)
        """

        # (batch_size, v1_len, 1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        # (batch_size, 1, v2_len)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).transpose(1, 2)

        # (batch_size, v1_len, v2_len)
        a = torch.bmm(v1, v2.transpose(1, 2))
        d = v1_norm * v2_norm
        attn = self._div_with_small_value(a, d)

        # (batch_size, v1_len, v2_len, hidden_size)
        attn_h = v2.unsqueeze(1) * attn.unsqueeze(3)
        # (batch_size, v1_len, hidden_size)
        attn_mean = self._div_with_small_value(attn_h.sum(dim=2), attn.sum(dim=2, keepdim=True))
        # (batch_size, v1_len, perspective_num)
        m_1 = self._full_matching(v1, attn_mean, w)

        attn_h_max, _ = attn_h.max(dim=2)
        m_2 = self._full_matching(v1, attn_h_max, w2)

        return m_1, m_2

    def _div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def forward(self, h1, h2, h1_mask, h2_mask):
        """
        :param h1: (batch_size, h1_len, hidden_size*2)
        :param h2:
        :param h1_mask: (batch_size, h1_len)
        :param h2_mask:
        :return: (batch_size, h1_len, perspective_num*8)
        """
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)

        # (batch_size, seq_len, hidden_size)
        h1_fw, h1_bw = torch.split(h1, self.hidden_size, dim=-1)
        h2_fw, h2_bw = torch.split(h2, self.hidden_size, dim=-1)

        # full-matching
        h2_fw_last_state = utils.get_last_state(h2_fw, h2_mask)
        h2_bw_last_state = h2_bw[:, 0, :]
        full_match_f = self._full_matching(h1_fw, h2_fw_last_state, self.fw_1)
        full_match_b = self._full_matching(h1_bw, h2_bw_last_state, self.bw_1)

        # maxpooling-matching
        maxpooling_match_f = self._maxpooling_matching(h1_fw, h2_fw, self.fw_2)
        maxpooling_match_b = self._maxpooling_matching(h1_bw, h2_bw, self.bw_2)

        # attention-matching, max-attention-matching
        attention_match_f, max_attention_match_f = self._attention_matching(h1_fw, h2_fw, self.fw_3, self.fw_4)
        attention_match_b, max_attention_match_b = self._attention_matching(h1_bw, h2_bw, self.bw_3, self.bw_4)

        # cat
        # (batch_size, h1_len, perspective_num*4)
        match_f = torch.cat([full_match_f, maxpooling_match_f, attention_match_f, max_attention_match_f], dim=2)
        match_b = torch.cat([full_match_b, maxpooling_match_b, attention_match_b, max_attention_match_b], dim=2)

        # cat
        # (batch_size, h1_len, perspective_num*8)
        match = torch.cat([match_f, match_b], dim=2)

        return match


class Aggregation(nn.Module):
    def __init__(self, perspective_num, hidden_size, dropout_p):
        super(Aggregation, self).__init__()

        self.perspective_num = perspective_num
        self.hidden_size = hidden_size

        self.rnn = encoder.Rnn(
            mode='LSTM',
            input_size=self.perspective_num*8,
            hidden_size=self.hidden_size,
            dropout_p=dropout_p,
            bidirectional=True,
            layer_num=1,
            is_bn=True,
        )

    def forward(self, q, a, q_mask, a_mask):
        """
        :param q: (batch_size, q_len, perspective_num*8)
        :param a:
        :param q_mask: (batch_size, q_len)
        :param a_mask:
        :return: (batch_size, hidden_size*4)
        """
        q = q.transpose(0, 1)  # (q_len, batch_size, perspective_num*8)
        a = a.transpose(0, 1)

        q = self.rnn(q, q_mask).transpose(0, 1)  # (batch_size, q_len, hidden_size*2)
        q_f, q_b = torch.split(q, self.hidden_size, dim=-1)
        q_final_state = utils.get_last_state(q_f, q_mask)  # (batch_size, hidden_size)
        q_init_state = q_b[:, 0, :]  # (batch_size, hidden_size)

        a = self.rnn(a, a_mask).transpose(0, 1)
        a_f, a_b = torch.split(a, self.hidden_size, dim=-1)
        a_final_state = utils.get_last_state(a_f, a_mask)
        a_init_state = a_b[:, 0, :]

        # (batch_size, hidden_size*4)
        state = torch.cat([q_final_state, q_init_state, a_final_state, a_init_state], dim=1)

        return state


class AnswerScore(nn.Module):
    def __init__(self, hidden_size, k, lamda, dropout_p):
        super(AnswerScore, self).__init__()

        self.hidden_size = hidden_size
        self.k = k
        self.lamda = lamda
        self.dropout_p = dropout_p

        self.w2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.w3 = nn.Linear(hidden_size*4, hidden_size*2)
        self.w4 = nn.Linear(hidden_size*4, hidden_size*4)
        self.w5 = nn.Linear(hidden_size*4, 1)

        self.rnn = nn.GRUCell(input_size=hidden_size*2, hidden_size=hidden_size*4)
        self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()

    def reset_parameters(self):
        """ use xavier_uniform to initialize GRU/LSTM weights"""
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, m, s1, s2, s3, m_mask):
        """
        :param m: (batch_size, m_len, hidden_size*2)
        :param s1: (batch_size, hidden_size*4)
        :param s2:
        :param s3:
        :param m_mask: (batch_size, m_len)
        :return: (batch_size, 3)
        """

        s = [s1, s2, s3]

        # for i in range(m.size(0)):
        #     if s1[i][0].item() == s2[i][0].item() == s3[i][0].item():
        #         print('s1', s1[i])
        #         print('s2', s2[i])
        #         print('s3', s3[i])




        result = []
        for i in range(3):
            st = s[i]

            for j in range(self.k):
                stt = self.w3(st).unsqueeze(1).expand(m.size())  # (batch_size, m_len, hidden_size*2)
                mtt = self.w2(m)  # (batch_size, m_lem, hidden_size*2)
                att = self.lamda * f.cosine_similarity(mtt, stt, dim=2)  # (batch_size, m_len)

                mask = m_mask.eq(0)
                att.masked_fill_(mask, -float('inf'))
                att = f.softmax(att, dim=1).unsqueeze(1)  # (batch_size, 1, m_len)

                f_att = torch.bmm(att, m).squeeze(1)  # (batch_size, hidden_size*2)
                f_att = self.dropout(f_att)

                st = self.rnn(f_att, st)

            st = self.dropout(st)
            st = f.relu(self.w4(st))
            st = self.dropout(st)
            st = self.w5(st).view(-1)  # (batch_size)
            result.append(st)

        result = torch.stack(result, dim=1)  # (batch, 3)
        result = f.log_softmax(result, dim=1)

        return result
