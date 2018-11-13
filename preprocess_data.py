# coding = utf-8
# author = xy

from data_pre import clean_data
from data_pre import wfqd
import json
import pandas as pd
import numpy as np
from rouge import Rouge
import sys
import os
import utils
import pickle
import time
import gensim
from config import config_base

config = config_base.config


# convert .json to .pandas
# return: df
def organize_data(file_path, is_start=True):
    data = []
    if is_start:
        with open(file_path, 'r', encoding='utf-8') as file:
            for sentence in file.readlines():
                d = json.loads(sentence)
                tmp = [d['query_id'], d['query'], d['passage'], d['alternatives']]
                if 'answer' in d:
                    tmp.append(d['answer'])
                data.append(tmp)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            result = json.load(file)

        for r in result:
            if r['flag']:
                tmp = [r['query_id'], r['query'], r['passage'], r['alternatives'], r['answer']]
                data.append(tmp)

    if len(data[0]) == 4:
        columns = ['query_id', 'query', 'passage', 'alternatives']
    else:
        columns = ['query_id', 'query', 'passage', 'alternatives', 'answer']

    df = pd.DataFrame(data=data, columns=columns)

    return df


# 数据预处理
def deal_data(df):
    """ 数据预处理: 全半角转换， 繁体->简体 """

    querys = df['query'].values
    passages = df['passage'].values
    querys = clean_data.deal_data(querys)
    passages = clean_data.deal_data(passages)
    df['query'] = querys
    df['passage'] = passages

    return df


# 分离选项
def split_alter(df, is_test=False):
    alternatives = df['alternatives']
    a_item = []
    b_item = []
    c_item = []
    flag = []
    for alter in alternatives:
        alter_list = alter.split('|')
        alter_list = [a.strip() for a in alter_list]
        if is_test is False:
            alter_set = set(alter_list)
            if len(alter_set) == 3 and '' not in alter_set and ('不确定' in alter_set or '不能确定' in alter_set or
                                                                '无发确定' in alter_set or '无法确定' in alter_set or
                                                                '无法确实' in alter_set or '无法确认' in alter_set):
                flag.append(True)
                a_item.append(alter_list[0])
                b_item.append(alter_list[1])
                c_item.append(alter_list[2])
            else:
                flag.append(False)
                a_item.append('xxxx')
                b_item.append('xxxx')
                c_item.append('xxxx')
        else:
            if len(alter_list) == 3:
                a = alter_list[0]
                if a == '':
                    a = alter_list[1]
                if a == '':
                    a = alter_list[2]
                if a == '':
                    a = 'xxx'

                if alter_list[0] == '':
                    a_item.append(a)
                else:
                    a_item.append(alter_list[0])

                if alter_list[1] == '':
                    b_item.append(a)
                else:
                    b_item.append(alter_list[1])

                if alter_list[2] == '':
                    c_item.append(a)
                else:
                    c_item.append(alter_list[2])
            elif len(alter_list) == 2:
                a = alter_list[0]
                if a == '':
                    a = alter_list[1]
                if a == '':
                    a = 'xxx'

                if alter_list[0] == '':
                    a_item.append(a)
                else:
                    a_item.append(alter_list[0])

                if alter_list[1] == '':
                    b_item.append(a)
                    c_item.append(a)
                else:
                    b_item.append(alter_list[1])
                    c_item.append(alter_list[1])
            elif len(alter_list) == 1:
                a = alter_list[0]
                if a == '':
                    a = 'xxx'

                if alter_list[0] == '':
                    a_item.append(a)
                    b_item.append(a)
                    c_item.append(a)
                else:
                    a_item.append(alter_list[0])
                    b_item.append(alter_list[0])
                    c_item.append(alter_list[0])

            flag.append(True)

    print('split alter, data num:%d/%d' % (sum(flag), len(flag)))

    df['a_item'] = a_item
    df['b_item'] = b_item
    df['c_item'] = c_item
    df['alter_flag'] = flag

    return df


# 将答案为wfqd的sample翻倍
def add_wfqd(df):
    answers = df['answer'].values
    flag = []
    for a in answers:
        if a.strip() == '无法确定':
            flag.append(True)
        else:
            flag.append(False)
    df['flag_wfqd'] = flag

    df_wfqd = df[df['flag_wfqd']].copy()
    df = pd.concat([df, df_wfqd], axis=0)
    print('double wfqd samples, total_num:%d, wfqd_num:%d, new_df_num:%d' % (len(answers), len(df_wfqd), len(df)))
    return df


# 长度限定
# p: 500, q: 30, a: 5
def jieduan(df):
    passages = df['passage'].values
    querys = df['query'].values
    a_item = df['a_item'].values
    b_item = df['b_item'].values
    c_item = df['c_item'].values

    # cut p
    flag_p = []
    for p in passages:
        p_list = utils.split_word(p.strip())
        if len(p_list) > 500:
            flag_p.append(False)
        else:
            flag_p.append(True)

    # cut q
    flag_q = []
    for q in querys:
        q_list = utils.split_word(q.strip())
        if len(q_list) > 30:
            flag_q.append(False)
        else:
            flag_q.append(True)

    # cut a_item
    flag_a = []
    for a in a_item:
        a_list = utils.split_word(a.strip())
        if len(a_list) > 5:
            flag_a.append(False)
        else:
            flag_a.append(True)

    # cut b_item
    flag_b = []
    for b in b_item:
        b_list = utils.split_word(b.strip())
        if len(b_list) > 5:
            flag_b.append(False)
        else:
            flag_b.append(True)

    # cut c_item
    flag_c = []
    for c in c_item:
        c_list = utils.split_word(c.strip())
        if len(c_list) > 5:
            flag_c.append(False)
        else:
            flag_c.append(True)

    assert len(flag_p) == len(flag_q) == len(flag_a) == len(flag_b) == len(flag_c)

    flag = []
    for fp, fq, fa, fb, fc in zip(flag_p, flag_q, flag_a, flag_b, flag_c):
        if fp and fq and fa and fb and fc:
            flag.append(True)
        else:
            flag.append(False)
    print('训练/验证集， 长度截断，保留数据_num:%d/%d, ratio:%.4f' % (sum(flag), len(flag), sum(flag)/len(flag)))

    df['jieduan_flag'] = flag

    return df


# 对于测试集， 对长度超过500的passage进行处理
def shorten_passage(df, max_len=500):

    sys.setrecursionlimit(1000000)
    rouge = Rouge(metrics=['rouge-l'])

    passages = df['passage'].values
    querys = df['query'].values
    p_tmp = []
    cc = 0
    for p, q in zip(passages, querys):
        p_list = utils.split_word(p)
        if len(p_list) <= max_len:
            p_tmp.append(p)
        else:
            cc += 1
            chuck_num = len(p_list) // max_len
            scores = []
            for i in range(chuck_num+1):
                pp = ''.join(p_list[i*max_len: (i+1)*max_len])
                score = rouge.get_scores(pp, q, avg=True)['rouge-l']['r']
                scores.append(score)
            index = np.argmax(scores)
            pp = ''.join(p_list[index*max_len: (index+1)*max_len])
            p_tmp.append(pp)
    df['passage'] = p_tmp
    print('shorten passage num: %d/%d' % (cc, len(df)))

    return df


# generate vocab based on 'data_gen/collect_txt'
def build_vocab_embedding(list_df, vocab_path, embedding_in_zh, embedding_in_en, embedding_out):
    data = []
    for df in list_df:
        if 'answer' in df:
            data = data + df[['query', 'passage', 'answer']].values.flatten().tolist()
        else:
            data = data + df[['query', 'passage']].values.flatten().tolist()

    vocab = set()
    for d in data:
        d_list = utils.split_word(d)
        for dd in d_list:
            vocab.add(dd)
    print('data, word_nums:%d' % len(vocab))

    # zh
    try:
        model_zh = gensim.models.KeyedVectors.load_word2vec_format(embedding_in_zh)
    except Exception as e:
        model_zh = gensim.models.KeyedVectors.load_word2vec_format(embedding_in_zh, binary=True, unicode_errors='ignore')

    tmp = set()
    for word in vocab:
        if word in model_zh:
            tmp.add(word)
    print('word_nums in pre-embedding:%d/%d, ratio:%.4f' % (len(tmp), len(vocab), len(tmp)/len(vocab)))

    # w2i = {'<pad>': 0, '<unk>': 1, ' ': 2}
    # i2w = {0: '<pad>', 1: '<unk>', 2: ' '}
    w2i = {'<pad>': 0}
    i2w = {0: '<pad>'}
    c = 1
    embedding = np.zeros([len(tmp) + 3, model_zh.vector_size])
    for word in tmp:
        w2i[word] = c
        i2w[c] = word
        if word in model_zh:
            embedding[c] = model_zh[word]
        c += 1
    w2i['<unk>'] = len(tmp) + 1
    i2w[len(tmp)+1] = '<unk>'
    w2i[' '] = len(tmp) + 2
    i2w[len(tmp)+2] = ' '
    lang = {'w2i': w2i, 'i2w': i2w}
    print('vacab length: %d' % (c+2))
    print('embedding size:', embedding.shape)

    # save
    with open(vocab_path, 'wb') as file:
        pickle.dump(lang, file)
    np.save(embedding_out, embedding)


# 生成 词性-index 表
def gen_tag_index(df):
    df = df[['query', 'passage']]
    data = df.values.flatten().tolist()
    tag2i = {'<pad>': 0, '<unk>': 1}
    cc = 2
    for d in data:
        _, tags = utils.split_word(d, have_tag=True)

        for t in tags:
            if t not in tag2i:
                tag2i[t] = cc
                cc += 1

    with open(config.tag_path, 'wb') as file:
        pickle.dump(tag2i, file)
    print('word flag num:%d' % len(tag2i))  # 98个


def gen_pre_file_for_train():
    if os.path.isfile(config.train_vocab_path) is False:
        time0 = time.time()
        print('gen train prepared file...')

        # 组织数据 json->df
        train_df = organize_data(config.train_data)
        train_df_1 = organize_data(config.train_data_1, is_start=False)
        train_df_2 = organize_data(config.train_data_2, is_start=False)
        train_df = pd.concat([train_df, train_df_1, train_df_2], axis=0)

        val_df = organize_data(config.val_data)

        # 预处理数据
        train_df = deal_data(train_df)
        val_df = deal_data(val_df)

        # vocab, embedding
        build_vocab_embedding(
            list_df=[train_df, val_df],
            vocab_path=config.train_vocab_path,
            embedding_in_zh=config.pre_embedding_zh,
            embedding_in_en=config.pre_embedding_en,
            embedding_out=config.train_embedding
        )

        # 生成词性表
        gen_tag_index(train_df)
        print('gen train prepared file, time:%d' % (time.time()-time0))


def gen_pre_file_for_test():
    if os.path.isfile(config.test_vocab_path) is False:
        time0 = time.time()
        print('gen test prepared file...')
        # 组织数据 json -> df
        test_df = organize_data(config.test_data)
        # 预处理数据
        test_df = deal_data(test_df)
        # vocab, embedding
        build_vocab_embedding(
            list_df=[test_df],
            vocab_path=config.test_vocab_path,
            embedding_in_zh=config.pre_embedding_zh,
            embedding_in_en=config.pre_embedding_en,
            embedding_out=config.test_embedding
        )
        print('gen test prepared file, time:%d' % (time.time()-time0))


def gen_train_val_datafile():
    if os.path.isfile(config.train_df) is False:
        print('gen train data...')
        time0 = time.time()

        df_1 = organize_data(config.train_data)
        # df_2 = organize_data(config.train_data_1, is_start=False)
        # df_3 = organize_data(config.train_data_2, is_start=False)
        # df = pd.concat([df_1, df_2], axis=0)
        df = df_1

        df = deal_data(df)
        # 分离选项
        df = split_alter(df)
        df = jieduan(df)
        df = df[df['alter_flag']]
        df = df[df['jieduan_flag']]
        df.to_csv(config.train_df, encoding='utf-8', index=False)
        print('gen train data, size:%d, time:%d' % (len(df), time.time()-time0))

    if os.path.isfile(config.val_df) is False:
        print('gen val data...')
        time0 = time.time()
        df = organize_data(config.val_data)
        df = deal_data(df)
        df = split_alter(df, is_test=True)
        df = shorten_passage(df)
        df.to_csv(config.val_df, encoding='utf-8', index=False)
        print('gen val data, size:%d, time:%d' % (len(df), time.time()-time0))


def gen_test_datafile():
    if os.path.isfile(config.test_df) is False:
        print('gen test data...')
        time0 = time.time()
        df = organize_data(config.test_data)
        df = deal_data(df)
        df = split_alter(df, is_test=True)
        df = shorten_passage(df)
        df.to_csv(config.test_df, encoding='utf-8', index=False)
        print('gen test data, size:%d, time:%d' % (len(df), time.time()-time0))


if __name__ == '__main__':
    pass




















