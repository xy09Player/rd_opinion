# encoding = utf-8
# author = xy

import utils
import torch
from torch import nn
import torch.nn.functional as f


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p, mode, gated_attention, is_bn):
        super(SelfAttention, self).__init__()
        self.mode = mode
        self.gated_attention = gated_attention
        self.is_bn = is_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.right_match = UniSelfAttention(input_size, hidden_size, mode, gated_attention, is_bn)
        self.left_match = UniSelfAttention(input_size, hidden_size, mode, gated_attention, is_bn)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, qt_aware, mask):
        """
        :param qt_aware: tensor (p_seq_len, batch_size, input_size)
        :param mask: tensor (batch_size, p_seq_len)
        :return: result (p_seq_len, batch_size, hidden_size*2)
        """
        qt_aware = self.dropout(qt_aware)

        right_result = self.right_match(qt_aware, mask)
        qt_aware_clip = utils.masked_flip(qt_aware, mask)
        left_result = self.left_match(qt_aware_clip, mask)
        left_result = utils.masked_flip(left_result, mask)

        result = torch.cat([right_result, left_result], dim=2)  # (p_seq_len, batch_size, hidden_size*2)
        result = mask.transpose(0, 1).unsqueeze(2) * result

        return result


class UniSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, mode, gated_attention, is_bn):
        super(UniSelfAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.gated_attention = gated_attention
        self.is_bn = is_bn

        if mode == 'LSTM':
            self.rnn_cell = nn.LSTMCell(
                input_size=input_size*2,
                hidden_size=hidden_size
            )
        elif mode == 'GRU':
            self.rnn_cell = nn.GRUCell(
                input_size=input_size*2,
                hidden_size=hidden_size
            )
        else:
            assert 1 == -1

        if gated_attention:
            self.gated_linear = nn.Linear(input_size*2, input_size*2)

        if is_bn:
            self.layer_norm = nn.LayerNorm(input_size*2)

        self.wp = nn.Linear(input_size, hidden_size)
        self.wpp = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

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

    def forward(self, qt_aware, content_mask):
        """
        :param qt_aware: tensor (p_seq_len, batch_size, input_size)
        :param content_mask: tensor (batch_size, p_seq_len)
        :return: hp, tensor (p_seq_len, batch_size, hidden_size)
        """

        batch_size = qt_aware.size(1)
        p_seq_len = qt_aware.size(0)

        h_0 = qt_aware.new_zeros(batch_size, self.hidden_size)
        h = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]

        wp = self.wp(qt_aware)  # (p_seq_len, batch_size, hidden_size)

        for t in range(p_seq_len):
            # attention
            pp = qt_aware[t]
            wpp = self.wpp(pp).unsqueeze(0)  # (1, batch_size, hidden_size)

            g = torch.tanh(wp + wpp)
            alpha = self.v(g).squeeze(2).transpose(0, 1)  # (batch_size, q_seq_len)

            # mask
            mask = content_mask.eq(0)
            alpha.masked_fill_(mask, -float('inf'))
            alpha = f.softmax(alpha, dim=1)

            h_alpha = torch.bmm(alpha.unsqueeze(1), qt_aware.transpose(0, 1)).squeeze(1)  # (batch_size, input_size)
            z = torch.cat([qt_aware[t], h_alpha], dim=1)

            if self.gated_attention:
                gate = torch.sigmoid(self.gated_linear(z))
                z = gate * z

            if self.is_bn:
                z = self.layer_norm(z)

            hp = self.rnn_cell(z, h[t])
            h.append(hp)

        if self.mode == 'LSTM':
            hp = [hh[0] for hh in h[1:]]
        elif self.mode == 'GRU':
            hp = [hh for hh in h[1:]]

        hp = torch.stack(hp)

        return hp
