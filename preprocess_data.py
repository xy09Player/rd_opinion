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
def organize_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for sentence in file.readlines():
            d = json.loads(sentence)
            tmp = [d['query_id'], d['query'], d['passage'], d['alternatives']]
            if 'answer' in d:
                tmp.append(d['answer'])
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


# 正例/负例 数据
def zheng_fu_li(df, is_test=False):
    wfqd_list = wfqd.wfqd_list
    alts = df['alternatives'].values
    alts = [alt.split('|') for alt in alts]
    zhengli = []
    fuli = []
    flag = []
    for alt_list in alts:
        if is_test:
            if len(alt_list) >= 2:
                a_tmp_1 = alt_list[0].strip()
                a_tmp_2 = alt_list[1].strip()
            elif len(alt_list) == 1:
                a_tmp_1 = alt_list[0].strip()
                a_tmp_2 = alt_list[0].strip()
            else:
                print('zheng_fu_li method, meet wrong data')
                a_tmp_1 = 'xxx'
                a_tmp_2 = 'xxx'
            if len(a_tmp_1) > len(a_tmp_2):
                a_tmp_1, a_tmp_2 = a_tmp_2, a_tmp_1

            if a_tmp_1 == '':
                a_tmp_1 = a_tmp_2

            zhengli.append(a_tmp_1)
            fuli.append(a_tmp_2)
            flag.append(True)

        else:
            flag_tmp = False
            for a in alt_list:
                if a.strip() in wfqd_list:
                    flag_tmp = True
                    break
            if flag_tmp is False:
                zhengli.append('xxx')
                fuli.append('xxx')
                flag.append(False)
            else:
                if len(alt_list) == 3:
                    if alt_list[0].strip() in wfqd_list:
                        a_tmp_1 = alt_list[1].strip()
                        a_tmp_2 = alt_list[2].strip()
                    elif alt_list[1].strip() in wfqd_list:
                        a_tmp_1 = alt_list[0].strip()
                        a_tmp_2 = alt_list[2].strip()
                    else:
                        a_tmp_1 = alt_list[0].strip()
                        a_tmp_2 = alt_list[1].strip()

                    if len(a_tmp_1) > len(a_tmp_2):
                        a_tmp_1, a_tmp_2 = a_tmp_2, a_tmp_1

                    if (a_tmp_1 in wfqd_list) or (a_tmp_2 in wfqd_list) or (a_tmp_1 == a_tmp_2) or (a_tmp_1 == '') or (a_tmp_2 == ''):
                        zhengli.append('xxx')
                        fuli.append('xxx')
                        flag.append(False)
                    else:
                        zhengli.append(a_tmp_1)
                        fuli.append(a_tmp_2)
                        flag.append(True)
                else:
                    zhengli.append('xxx')
                    fuli.append('xxx')
                    flag.append(False)

    assert len(zhengli) == len(fuli) == len(flag)
    if is_test is False:
        print('训练/验证集，正负例， 保留样本_num:%d/%d, ratio:%.4f' % (sum(flag), len(flag), sum(flag)/len(flag)))

    df['zhengli'] = zhengli
    df['fuli'] = fuli
    df['zf_flag'] = flag
    df['wfqd'] = '无法确定'
    return df


# 长度限定
# p: 500, q: 30, a: 5
def jieduan(df):
    passages = df['passage'].values
    querys = df['query'].values
    zhenglis = df['zhengli'].values
    fulis = df['fuli'].values

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

    # cut zhengli
    flag_z = []
    for z in zhenglis:
        z_list = utils.split_word(z.strip())
        if len(z_list) > 5:
            flag_z.append(False)
        else:
            flag_z.append(True)

    # cut fuli
    flag_f = []
    for f in fulis:
        f_list = utils.split_word(f.strip())
        if len(f_list) > 5:
            flag_f.append(False)
        else:
            flag_f.append(True)

    assert len(flag_p) == len(flag_q) == len(flag_z) == len(flag_f)

    flag = []
    for fp, fq, fz, ff in zip(flag_p, flag_q, flag_z, flag_f):
        if fp and fq and fz and ff:
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
        df = organize_data(config.train_data)
        df = deal_data(df)
        df = zheng_fu_li(df, is_test=False)
        df = jieduan(df)
        df = df[df['zf_flag']]
        df = df[df['jieduan_flag']]
        df.to_csv(config.train_df, index=False)
        print('gen train data, size:%d, time:%d' % (len(df), time.time()-time0))

    if os.path.isfile(config.val_df) is False:
        print('gen val data...')
        time0 = time.time()
        df = organize_data(config.val_data)
        df = deal_data(df)
        df = zheng_fu_li(df, is_test=False)
        df = shorten_passage(df)
        df = df[df['zf_flag']]
        df.to_csv(config.val_df, index=False)
        print('gen val data, size:%d, time:%d' % (len(df), time.time()-time0))


def gen_test_datafile():
    if os.path.isfile(config.test_df) is False:
        print('gen test data...')
        time0 = time.time()
        df = organize_data(config.test_data)
        df = deal_data(df)
        df = zheng_fu_li(df, is_test=True)
        df = shorten_passage(df)
        df.to_csv(config.test_df, index=False)
        print('gen test data, size:%d, time:%d' % (len(df), time.time()-time0))


if __name__ == '__main__':
    pass




















