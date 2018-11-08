# coding = utf-8
# author = xy

import torch
from torch.utils import data
import pandas as pd
import utils
import pickle
from data_pre import wfqd


class CustomDataset(data.Dataset):

    def __init__(self, df_file, vocab_path, tag_path, p_max_len=500, q_max_len=30, a_max_len=5, is_test=False):
        self.is_test = is_test
        # load
        df = pd.read_csv(df_file)
        passages = df['passage'].values.tolist()
        querys = df['query'].values.tolist()
        zhenglis = df['zhengli'].values.tolist()
        fulis = df['fuli'].values.tolist()
        wfqds = df['wfqd'].values.tolist()
        wfqd_list = wfqd.wfqd_list

        if 'answer' in df:
            answers = df['answer'].values.tolist()
            answer_tmp = []
            for answer, zhengli, fuli in zip(answers, zhenglis, fulis):
                if answer.strip() == zhengli:
                    answer_tmp.append(0)
                elif answer.strip() == fuli:
                    answer_tmp.append(1)
                elif answer.strip() in wfqd_list:
                    answer_tmp.append(2)
                else:
                    print('build dataset, meet wrong data, answer:%s, zhengli:%s, fuli:%s' % (answer, zhengli, fuli))
            self.answer_index = answer_tmp
        else:
            self.answer_index = None

        # word seg, flags, is_in
        self.p_word_list, self.p_tag_list, self.p_in, self.q_word_list, self.q_tag_list, self.q_in = \
            utils.deal_data(passages, querys)

        self.zhengli_index, self.zhengli_tag, self.zhengli_in = utils.deal_answer(zhenglis, passages)
        self.fuli_index, self.fuli_tag, self.fuli_in = utils.deal_answer(fulis, passages)
        self.wfqd_index, self.wfqd_tag, self.wfqd_in = utils.deal_answer(wfqds, passages)

        # vocab
        with open(vocab_path, 'rb') as file:
            lang = pickle.load(file)
            self.w2i = lang['w2i']

        # tag
        with open(tag_path, 'rb') as file:
            self.lang = pickle.load(file)

        self.p_max_len = p_max_len
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len

    def __getitem__(self, item):
        # 正常属性
        p_word_list = self.p_word_list[item]
        p_tag_list = self.p_tag_list[item]
        p_in = self.p_in[item]
        assert len(p_word_list) == len(p_tag_list) == len(p_in)

        q_word_list = self.q_word_list[item]
        q_tag_list = self.q_tag_list[item]
        q_in = self.q_in[item]
        assert len(q_word_list) == len(q_tag_list) == len(q_in)

        zhengli_word_list = self.zhengli_index[item]
        zhengli_tag_list = self.zhengli_tag[item]
        zhengli_in = self.zhengli_in[item]
        assert len(zhengli_word_list) == len(zhengli_tag_list) == len(zhengli_in)

        fuli_word_list = self.fuli_index[item]
        fuli_tag_list = self.fuli_tag[item]
        fuli_in = self.fuli_in[item]
        assert len(fuli_word_list) == len(fuli_tag_list) == len(fuli_in)

        wfqd_word_list = self.wfqd_index[item]
        wfqd_tag_list = self.wfqd_tag[item]
        wfqd_in = self.wfqd_in[item]
        assert len(wfqd_word_list) == len(wfqd_tag_list) == len(wfqd_in)

        if self.answer_index is not None:
            answer = self.answer_index[item]

        # index
        p_word_list = [self.w2i.get(word, self.w2i['<unk>']) for word in p_word_list]
        q_word_list = [self.w2i.get(word, self.w2i['<unk>']) for word in q_word_list]
        zhengli_word_list = [self.w2i.get(word, self.w2i['<unk>']) for word in zhengli_word_list]
        fuli_word_list = [self.w2i.get(word, self.w2i['<unk>']) for word in fuli_word_list]
        wfqd_word_list = [self.w2i.get(word, self.w2i['<unk>']) for word in wfqd_word_list]

        p_tag_list = [self.lang.get(tag, self.lang['<unk>']) for tag in p_tag_list]
        q_tag_list = [self.lang.get(tag, self.lang['<unk>']) for tag in q_tag_list]
        zhengli_tag_list = [self.lang.get(tag, self.lang['<unk>']) for tag in zhengli_tag_list]
        fuli_tag_list = [self.lang.get(tag, self.lang['<unk>']) for tag in fuli_tag_list]
        wfqd_tag_list = [self.lang.get(tag, self.lang['<unk>']) for tag in wfqd_tag_list]

        # padding
        p_word_list = self.__pad__(p_word_list, self.p_max_len, self.w2i['<pad>'])
        p_tag_list = self.__pad__(p_tag_list, self.p_max_len, self.lang['<pad>'])
        p_in = self.__pad__(p_in, self.p_max_len, 0)

        q_word_list = self.__pad__(q_word_list, self.q_max_len, self.w2i['<pad>'])
        q_tag_list = self.__pad__(q_tag_list, self.q_max_len, self.lang['<pad>'])
        q_in = self.__pad__(q_in, self.q_max_len, 0)

        zhengli_word_list = self.__pad__(zhengli_word_list, self.a_max_len, self.w2i['<pad>'])
        zhengli_tag_list = self.__pad__(zhengli_tag_list, self.a_max_len, self.lang['<pad>'])
        zhengli_in = self.__pad__(zhengli_in, self.a_max_len, 0)

        fuli_word_list = self.__pad__(fuli_word_list, self.a_max_len, self.w2i['<pad>'])
        fuli_tag_list = self.__pad__(fuli_tag_list, self.a_max_len, self.lang['<pad>'])
        fuli_in = self.__pad__(fuli_in, self.a_max_len, 0)

        wfqd_word_list = self.__pad__(wfqd_word_list, self.a_max_len, self.w2i['<pad>'])
        wfqd_tag_list = self.__pad__(wfqd_tag_list, self.a_max_len, self.lang['<pad>'])
        wfqd_in = self.__pad__(wfqd_in, self.a_max_len, 0)

        # tensor
        p_word_list = torch.LongTensor(p_word_list)
        p_tag_list = torch.LongTensor(p_tag_list)
        p_in = torch.LongTensor(p_in)

        q_word_list = torch.LongTensor(q_word_list)
        q_tag_list = torch.LongTensor(q_tag_list)
        q_in = torch.LongTensor(q_in)

        zhengli_word_list = torch.LongTensor(zhengli_word_list)
        zhengli_tag_list = torch.LongTensor(zhengli_tag_list)
        zhengli_in = torch.LongTensor(zhengli_in)

        fuli_word_list = torch.LongTensor(fuli_word_list)
        fuli_tag_list = torch.LongTensor(fuli_tag_list)
        fuli_in = torch.LongTensor(fuli_in)

        wfqd_word_list = torch.LongTensor(wfqd_word_list)
        wfqd_tag_list = torch.LongTensor(wfqd_tag_list)
        wfqd_in = torch.LongTensor(wfqd_in)

        if self.answer_index is not None:
            answer = torch.LongTensor([answer])

        if self.is_test is False:
            return p_word_list, p_tag_list, p_in, \
                   q_word_list, q_tag_list, q_in, \
                   zhengli_word_list, zhengli_tag_list, zhengli_in, \
                   fuli_word_list, fuli_tag_list, fuli_in, \
                   wfqd_word_list, wfqd_tag_list, wfqd_in, \
                   answer

        else:
            return p_word_list, p_tag_list, p_in, \
                   q_word_list, q_tag_list, q_in, \
                   zhengli_word_list, zhengli_tag_list, zhengli_in, \
                   fuli_word_list, fuli_tag_list, fuli_in, \
                   wfqd_word_list, wfqd_tag_list, wfqd_in

    def __len__(self):
        return len(self.p_word_list)

    def __pad__(self, index_list, max_len, pad):
        if len(index_list) <= max_len:
            index_list = index_list + [pad] * (max_len - len(index_list))
        else:
            index_list = index_list[: max_len]
        return index_list
