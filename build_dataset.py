# coding = utf-8
# author = xy

import torch
from torch.utils import data
import pandas as pd
import utils
import json
import codecs
import os
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

        self.zhengli_index = [utils.split_word(zhengli) for zhengli in zhenglis]
        self.fuli_index = [utils.split_word(fuli) for fuli in fulis]
        self.wfqd_index = [utils.split_word(www) for www in wfqds]

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

        # EMLo数据
        with open('ELMo/configs/cnn_50_100_512_4096_sample.json', 'r') as fin:
            self.config = json.load(fin)
        self.max_char_len = self.config['token_embedder']['max_characters_per_token']

        # For the model trained with character-based word encoder.
        if self.config['token_embedder']['char_dim'] > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join('ELMo/zhs.model', 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)

        # For the model trained with word form word encoder.
        if self.config['token_embedder']['word_dim'] > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join('ELMo/zhs.model', 'word.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)

    def __getitem__(self, item):
        # 正常属性
        p_word_list = self.p_word_list[item]
        p_tag_list = self.p_tag_list[item]
        p_in = self.p_in[item]
        q_word_list = self.q_word_list[item]
        q_tag_list = self.q_tag_list[item]
        q_in = self.q_in[item]

        zhengli_word_list = self.zhengli_index[item]
        fuli_word_list = self.fuli_index[item]
        wfqd_word_list = self.wfqd_index[item]
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

        # padding
        p_word_list = self.__pad__(p_word_list, self.p_max_len, self.w2i['<pad>'])
        p_tag_list = self.__pad__(p_tag_list, self.p_max_len, self.lang['<pad>'])
        p_in = self.__pad__(p_in, self.p_max_len, 0)
        q_word_list = self.__pad__(q_word_list, self.q_max_len, self.w2i['<pad>'])
        q_tag_list = self.__pad__(q_tag_list, self.q_max_len, self.lang['<pad>'])
        q_in = self.__pad__(q_in, self.q_max_len, 0)
        zhengli_word_list = self.__pad__(zhengli_word_list, self.a_max_len, self.w2i['<pad>'])
        fuli_word_list = self.__pad__(fuli_word_list, self.a_max_len, self.w2i['<pad>'])
        wfqd_word_list = self.__pad__(wfqd_word_list, self.a_max_len, self.w2i['<pad>'])

        # tensor
        p_word_list = torch.LongTensor(p_word_list)
        p_tag_list = torch.LongTensor(p_tag_list)
        p_in = torch.LongTensor(p_in)
        q_word_list = torch.LongTensor(q_word_list)
        q_tag_list = torch.LongTensor(q_tag_list)
        q_in = torch.LongTensor(q_in)
        zhengli_word_list = torch.LongTensor(zhengli_word_list)
        fuli_word_list = torch.LongTensor(fuli_word_list)
        wfqd_word_list = torch.LongTensor(wfqd_word_list)
        if self.answer_index is not None:
            answer = torch.LongTensor([answer])

        # Elmo
        p_word_list_elmo = self.p_word_list[item]
        q_word_list_elmo = self.q_word_list[item]
        zhengli_word_list_elmo = self.zhengli_index[item]
        fuli_word_list_elmo = self.fuli_index[item]
        wfqd_word_list_elmo = self.wfqd_index[item]

        p_word_elmo, p_char_elmo = self.__gen_elmo__(p_word_list_elmo, self.p_max_len)
        q_word_elmo, q_char_elmo = self.__gen_elmo__(q_word_list_elmo, self.q_max_len)
        zhengli_word_elmo, zhengli_char_elmo = self.__gen_elmo__(zhengli_word_list_elmo, self.a_max_len)
        fuli_word_elmo, fuli_char_elmo = self.__gen_elmo__(fuli_word_list_elmo, self.a_max_len)
        wfqd_word_elmo, wfqd_char_elmo = self.__gen_elmo__(wfqd_word_list_elmo, self.a_max_len)

        # tensor
        p_word_elmo = torch.LongTensor(p_word_elmo)
        p_char_elmo = torch.LongTensor(p_char_elmo)

        q_word_elmo = torch.LongTensor(q_word_elmo)
        q_char_elmo = torch.LongTensor(q_char_elmo)

        zhengli_word_elmo = torch.LongTensor(zhengli_word_elmo)
        zhengli_char_elmo = torch.LongTensor(zhengli_char_elmo)

        fuli_word_elmo = torch.LongTensor(fuli_word_elmo)
        fuli_char_elmo = torch.LongTensor(fuli_char_elmo)

        wfqd_word_elmo = torch.LongTensor(wfqd_word_elmo)
        wfqd_char_elmo = torch.LongTensor(wfqd_char_elmo)

        if self.is_test is False:
            return p_word_list, p_tag_list, p_in, q_word_list, q_tag_list, q_in, \
                   zhengli_word_list, fuli_word_list, wfqd_word_list, \
                   p_word_elmo, p_char_elmo, q_word_elmo, q_char_elmo, zhengli_word_elmo, zhengli_char_elmo, \
                   fuli_word_elmo, fuli_char_elmo, wfqd_word_elmo, wfqd_char_elmo, answer

        else:
            return p_word_list, p_tag_list, p_in, q_word_list, q_tag_list, q_in, \
                   zhengli_word_list, fuli_word_list, wfqd_word_list, \
                   p_word_elmo, p_char_elmo, q_word_elmo, q_char_elmo, zhengli_word_elmo, zhengli_char_elmo, \
                   fuli_word_elmo, fuli_char_elmo, wfqd_word_elmo, wfqd_char_elmo

    def __len__(self):
        return len(self.p_word_list)

    def __pad__(self, index_list, max_len, pad):
        if len(index_list) <= max_len:
            index_list = index_list + [pad] * (max_len - len(index_list))
        else:
            index_list = index_list[: max_len]
        return index_list

    def __gen_elmo__(self, word_list, max_len):
        word_list = ['<bos>'] + list(word_list) + ['<eos>']

        word_elmo_list = [self.word_lexicon.get(word, self.word_lexicon['<oov>']) for word in word_list]
        word_elmo_list = self.__pad__(word_elmo_list, max_len+2, self.word_lexicon['<pad>'])

        char_elmo_list = []
        for word in word_list:
            if len(word) + 2 > self.max_char_len:
                word = word[: self.max_char_len-2]
            tmp = [self.char_lexicon['<eow>']]
            if word == '<bos>' or word == '<eos>':
                tmp.append(self.char_lexicon[word])
                tmp.append(self.char_lexicon['<bow>'])
            else:
                for cc in word:
                    tmp.append(self.char_lexicon.get(cc, self.char_lexicon['<oov>']))
                tmp.append(self.char_lexicon['<bow>'])
            tmp = tmp + [self.char_lexicon['<pad>']]*(self.max_char_len-len(tmp))
            char_elmo_list.append(tmp)
        char_elmo_list = char_elmo_list + [[self.char_lexicon['<pad>']]*self.max_char_len]*(max_len+2-len(char_elmo_list))

        return word_elmo_list, char_elmo_list









