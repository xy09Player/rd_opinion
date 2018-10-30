# encoding = utf-8
# author = xy

from data_pre import wfqd
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import utils
import pickle


def load_w2v(embedding_path):
    """ load embedding vector """
    embedding_np = np.load(embedding_path)
    return embedding_np


def load_data(df_file, vocab_path, tag_path, q_max_len=30, p_max_len=500, a_max_len=5):
    """
    load data from .csv
    # 1. load
    # 2. index, tag(词性), 是否在答案中出现， 是否是标题
    # 3. padding
    return: content, question, zhengli, fuli, wfqd, answer
    """

    # load
    df = pd.read_csv(df_file)
    querys = df['query'].values.tolist()
    passages = df['passage'].values.tolist()
    zhenglis = df['zhengli'].values.tolist()
    fulis = df['fuli'].values.tolist()
    wfqds = df['wfqd'].values.tolist()
    wfqd_list = wfqd.wfqd_list

    if 'answer' in df:
        answers = df['answer'].values.tolist()
        answers_tmp = []
        for answer, zhengli, fuli in zip(answers, zhenglis, fulis):
            if answer.strip() == zhengli:
                answers_tmp.append(0)
            elif answer.strip() == fuli:
                answers_tmp.append(1)
            elif answer.strip() in wfqd_list:
                answers_tmp.append(2)
            else:
                print('load_data, meet wrong data, answer:%s, zhengli:%s, fuli:%s' % (answer, zhengli, fuli))

    # words, flags, is_in
    q_index, q_tag, q_in, p_index, p_tag, p_in = utils.deal_data(querys, passages)
    zhengli_index = [utils.split_word(zhengli) for zhengli in zhenglis]
    fuli_index = [utils.split_word(fuli) for fuli in fulis]
    wfqd_index = [utils.split_word(w) for w in wfqds]

    # words -> index
    q_index = utils.words2index(q_index, vocab_path)
    p_index = utils.words2index(p_index, vocab_path)
    zhengli_index = utils.words2index(zhengli_index, vocab_path)
    fuli_index = utils.words2index(fuli_index, vocab_path)
    wfqd_index = utils.words2index(wfqd_index, vocab_path)

    # flags -> index
    q_tag = utils.tags2index(q_tag, tag_path)
    p_tag = utils.tags2index(p_tag, tag_path)

    # padding
    q_index = utils.pad(q_index, q_max_len)
    q_tag = utils.pad(q_tag, q_max_len)
    q_in = utils.pad(q_in, q_max_len)

    p_index = utils.pad(p_index, p_max_len)
    p_tag = utils.pad(p_tag, p_max_len)
    p_in = utils.pad(p_in, p_max_len)

    zhengli_index = utils.pad(zhengli_index, a_max_len)
    fuli_index = utils.pad(fuli_index, a_max_len)
    wfqd_index = utils.pad(wfqd_index, a_max_len)

    if 'answer' in df:
        return [p_index, p_tag, p_in, q_index, q_tag, q_in, zhengli_index, fuli_index, wfqd_index, answers_tmp]
    else:
        return [p_index, p_tag, p_in, q_index, q_tag, q_in, zhengli_index, wfqd_index, fuli_index]


def build_loader(dataset, batch_size, shuffle, drop_last):
    """
    build data loader
    return: a instance of Dataloader
    """
    dataset = [torch.LongTensor(d) for d in dataset]
    dataset = data.TensorDataset(*dataset)
    data_iter = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter













