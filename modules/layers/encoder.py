# coding = utf-8
# author = xy

import torch
from torch import nn


class Rnn(nn.Module):
    def __init__(self, mode, input_size, hidden_size, dropout_p, bidirectional, layer_num, is_bn, batch_first=False):
        super(Rnn, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.direction_num = bidirectional
        self.layer_num = layer_num
        self.is_bn = is_bn

        if mode == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                dropout=dropout_p if layer_num > 1 else 0
            )
        elif mode == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                dropout=dropout_p if layer_num > 1 else 0
            )
        if is_bn:
            self.layer_norm = nn.LayerNorm(input_size)
        self.drop = nn.Dropout(p=dropout_p)

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

    def forward(self, vec, mask, need_final_state=False):
        """
        :param vec: tensor (seq_len, batch_size, input_size)
        :param mask: tensor (batch_size, seq_len)
        :return: outputs: tensor (seq_len, batch_size, hidden_size*bidirectional)
        """

        # layer normalization
        if self.is_bn:
            seq_len, batch_size, input_size = vec.shape
            vec = vec.contiguous().view(-1, input_size)
            vec = self.layer_norm(vec)
            vec = vec.view(seq_len, batch_size, input_size)

        # dropout
        vec = self.drop(vec)

        # 这种方式
        lengths = mask.long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, descending=True)
        _, idx_unsort = torch.sort(idx_sort)

        v_sort = vec.index_select(1, idx_sort)
        v_pack = nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        outputs, state = self.rnn(v_pack, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.index_select(1, idx_unsort)

        # 获得最后状态
        if need_final_state:
            # state = state.index_select(1, idx_unsort)

            len_dix = (lengths-1).view(-1, 1).expand(-1, outputs.size(2)).unsqueeze(0)
            state = outputs.gather(0, len_dix)
            state = state.squeeze(0)

        # 填充
        max_len = torch.max(lengths).item()
        seq_len = vec.size(0)
        if max_len != seq_len:
            pad_len = seq_len - max_len
            batch_size = vec.size(1)
            hidden_size = self.hidden_size * 2 if self.direction_num else self.hidden_size
            padding = outputs.new_zeros(pad_len, batch_size, hidden_size)
            outputs = torch.cat([outputs, padding], dim=0)

        if need_final_state:
            return outputs, state
        else:
            return outputs
