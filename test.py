# coding = utf-8
# author = xy

import os
import sys
import time
import pickle
import pandas as pd
import torch
from torch import nn
from data_pre import wfqd
import loader
import numpy as np
import build_dataset
import utils
import preprocess_data
from config import config_base
from config import config_m_reader
from config import config_m_reader_plus
from config import config_ensemble

from modules import m_reader


def test(config):
    time0 = time.time()

    # prepare
    preprocess_data.gen_pre_file_for_test()

    # load w2v
    embedding_np_train = loader.load_w2v(config.train_embedding + '.npy')
    embedding_np_test = loader.load_w2v(config.test_embedding + '.npy')

    # prepare: test_df
    if config.is_true_test:
        preprocess_data.gen_test_datafile()

    # load data
    if config.is_true_test is False:
        if os.path.isfile(config.val_true_pkl):
            with open(config.val_true_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = build_dataset.CustomDataset(
                df_file=config.val_df,
                vocab_path=config.train_vocab_path,
                tag_path=config.tag_path,
                is_test=True
            )
            with open(config.val_true_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    else:
        if os.path.isfile(config.test_pkl):
            with open(config.test_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = build_dataset.CustomDataset(
                df_file=config.test_df,
                vocab_path=config.test_vocab_path,
                tag_path=config.tag_path,
                is_test=True
            )
            with open(config.test_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    # model initial
    param = {
        'embedding': embedding_np_train,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': config.encoder_dropout_p,
        'encoder_bidirectional': config.encoder_bidirectional,
        'encoder_layer_num': config.encoder_layer_num,
        'is_bn': config.is_bn,
        'k': config.k,
        'num_align_hops': config.num_align_hops
    }

    model = eval(config.model_name).Model(param)

    # load model param, and training state
    model_path = os.path.join('model', config.model_test)
    print('load model, ', model_path)
    state = torch.load(model_path)
    model.load_state_dict(state['best_model_state'])

    # 改变embedding_fix
    if config.is_true_test:
        model.embedding.sd_embedding.embedding_fix = nn.Embedding(
            num_embeddings=embedding_np_test.shape[0],
            embedding_dim=embedding_np_test.shape[1],
            padding_idx=0,
            _weight=torch.Tensor(embedding_np_test)
        )
        model.embedding.sd_embedding.embedding_fix.weight.requires_grad = False
        model.embedding.sd_embedding.vocab_size = embedding_np_test.shape[0]
    model = model.cuda()

    best_loss = state['best_loss']
    best_val_accuracy = state['best_val_accuracy']
    best_epoch = state['best_epoch']
    best_step = state['best_step']
    best_time = state['best_time']
    use_time = state['time']
    print('best_epoch:%2d, best_step:%5d, best_loss:%.4f, val_accuracy:%.4f, best_time:%d, use_time:%d' %
          (best_epoch, best_step, best_loss, best_val_accuracy, best_time, use_time))

    # gen result
    result = []
    result_range = []

    model.eval()
    with torch.no_grad():
        cc = 0
        cc_total = len(test_loader)
        print('total iter_num:%d' % cc_total)
        for batch in test_loader:
            # cuda, cut
            batch = utils.deal_batch(batch)
            outputs = model(batch)  # (batch_size, 3)
            _, k = torch.max(outputs, dim=1)
            k = k.cpu().numpy().tolist()
            result = result + k

            outputs = outputs.cpu().numpy().tolist()
            result_range = result_range + outputs

            cc += 1
            if cc % 100 == 0:
                print('processing: %d/%d' % (cc, cc_total))

    if config.is_true_test:
        df = pd.read_csv(config.test_df, encoding='utf-8')
    else:
        df = pd.read_csv(config.val_df, encoding='utf-8')

    # 生成结果
    a_items = df['a_item'].values
    b_items = df['b_item'].values
    c_items = df['c_item'].values
    alts = df['alternatives'].values
    tmp = []
    for r, a, b, c, alt in zip(result, a_items, b_items, c_items, alts):
        alt_list = alt.split('|')
        if r == 0:
            if a == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif a == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif a == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==0, meet wrong data')
        elif r == 1:
            if b == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif b == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif b == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==1, meet wrong data')
        else:
            if c == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif c == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif c == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==2, meet wrong data')

    # gen a submission
    if config.is_true_test:
        query_ids = df['query_id']
        with open(config.submission, 'w') as file:
            for i, r in zip(query_ids, tmp):
                file.writelines(str(i) + '\t' + r + '\n')

    # my_metrics
    if config.is_true_test is False:
        answers = df['answer']
        flag = []
        for a, r in zip(answers, tmp):
            if a == r:
                flag.append(True)
            else:
                flag.append(False)
        print('accuracy:%.4f' % (sum(flag)/len(answers)))

    # to .csv
    if config.is_true_test is False:
        df['answer_pred'] = tmp
        df = df[['query_id', 'query', 'passage', 'alternatives', 'answer', 'answer_pred']]
        csv_path = os.path.join('result', config.model_test+'_val.csv')
        df.to_csv(csv_path, index=False)

    # save result_ans_range
    if config.is_true_test:
        save_path = os.path.join('result/ans_range', config.model_test+'_submission.pkl')
    else:
        save_path = os.path.join('result/ans_range', config.model_test+'_val.pkl')
    torch.save(result_range, save_path)
    print('time:%d' % (time.time()-time0))


def test_ensemble(config):
    time0 = time.time()

    if config.is_true_test:
        df = pd.read_csv(config.test_df, encoding='utf-8')
    else:
        df = pd.read_csv(config.val_df, encoding='utf-8')

    # 投票
    model_lst = config.model_lst
    result_toupiaos = []
    for ml in model_lst:
        result_path = os.path.join('result/ans_range', ml)
        ans_range = torch.load(result_path)
        ans_range = np.array(ans_range)
        result_tmp = np.argmax(ans_range, axis=1)
        result_toupiaos.append(result_tmp)

    # 概率修正
    # ans_dis = 'data_gen/ans_dis.pkl'
    # with open(ans_dis, 'rb') as file:
    #     ans_dis = pickle.load(file)
    #
    # zhenglis = df['zhengli'].values
    # fulis = df['fuli'].values
    # alts = df['alternatives'].values
    # wfqd_list = wfqd.wfqd_list
    # ans_dis_1 = {}
    # num = 0
    # for i in range(len(df)):
    #     flag = True
    #     value = result_toupiaos[0][i]
    #     for j in range(len(model_lst)):
    #         if result_toupiaos[j][i] != value:
    #             flag = False
    #             break
    #     if flag:
    #         num += 1
    #         alt_list = alts[i].split('|')
    #         if value == 0:
    #             word = zhenglis[i]
    #         elif value == 1:
    #             word = fulis[i]
    #         else:
    #             if alt_list[0].strip() in wfqd_list:
    #                 word = alt_list[0].strip()
    #             elif alt_list[1].strip() in wfqd_list:
    #                 word = alt_list[1].strip()
    #             else:
    #                 word = alt_list[2].strip()
    #
    #         if word in ans_dis_1:
    #             ans_dis_1[word] += 1
    #         else:
    #             ans_dis_1[word] = 1
    # # 获取投票一致的word表
    # ans_dis_1_tmp = {}
    # for k, v in ans_dis_1.items():
    #     ans_dis_1_tmp[k] = v/len(df)
    # ans_dis_1 = ans_dis_1_tmp
    #
    # # 获取投票不一致的word表（先验）
    # num_ratio = 1 - num / len(df)
    # ans_dis_2_x = {}
    # for k, v in ans_dis.items():
    #     ans_dis_2_x[k] = v * num_ratio
    #
    # # 获取投票不一致的word（后验）
    # ans_dis_2_h = {}
    # for k, v in ans_dis.items():
    #     if k in ans_dis_1:
    #         ans_dis_2_h[k] = v - ans_dis_1[k]
    #     else:
    #         ans_dis_2_h[k] = v
    #
    # # 获取修正概率
    # ans_dis_2 = {}
    # for k, v in ans_dis_2_x.items():
    #     if k in ans_dis_2_h:
    #         ans_dis_2[k] = ans_dis_2_h[k] / v

    # 加权求和： 待修改
    model_lst = config.model_lst
    model_weight = config.model_weight
    result_jiaquan = np.zeros([len(df), 3])
    for ml, mw in zip(model_lst, model_weight):
        result_path = os.path.join('result/ans_range', ml)
        ans_range = torch.load(result_path)
        ans_range = np.array(ans_range)

        exp_ans_range = np.exp(ans_range)
        sum_ans_range = np.sum(exp_ans_range, axis=1).reshape(len(exp_ans_range), -1)
        ans_range = exp_ans_range/sum_ans_range

        ans_range = ans_range * ans_range
        result_jiaquan += ans_range * mw
    result_jiaquan = np.argmax(result_jiaquan, axis=1)

    # 整合
    result = []
    r_flag = []
    for i in range(len(result_jiaquan)):
        flag = True
        value = result_toupiaos[0][i]
        for j in range(len(model_lst)):
            if result_toupiaos[j][i] != value:
                flag = False
                break
        if flag:
            r_flag.append('toupiao')
            result.append(value)
        else:
            # vec = result_jiaquan[i]
            # vec[0] = vec[0] * ans_dis_2.get(zhenglis[i], 1)
            # vec[1] = vec[1] * ans_dis_2.get(fulis[i], 1)
            # vec[2] = vec[2] * ans_dis_2.get('无法确定', 1)
            # r = np.argmax(vec, axis=0)
            # r_flag.append('jiaquan')
            # result.append(r)
            r_flag.append('jiaquan')
            result.append(result_jiaquan[i])

    # 生成结果
    a_items = df['a_item'].values
    b_items = df['b_item'].values
    c_items = df['c_item'].values
    alts = df['alternatives'].values
    tmp = []
    for r, a, b, c, alt in zip(result, a_items, b_items, c_items, alts):
        alt_list = alt.split('|')
        if r == 0:
            if a == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif a == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif a == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==0, meet wrong data')
        elif r == 1:
            if b == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif b == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif b == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==1, meet wrong data')
        else:
            if c == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif c == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif c == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==2, meet wrong data')

    # gen a submission
    if config.is_true_test:
        query_ids = df['query_id']
        with open(config.submission, 'w') as file:
            for i, r in zip(query_ids, tmp):
                file.writelines(str(i) + '\t' + r + '\n')

    # my_metrics
    if config.is_true_test is False:
        answers = df['answer']
        flag = []
        for a, r in zip(answers, tmp):
            if a == r:
                flag.append(True)
            else:
                flag.append(False)
        print('accuracy:%.4f' % (sum(flag)/len(answers)))

    # to .csv
    if config.is_true_test is False:
        df['answer_pred'] = tmp
        df['r_flag'] = r_flag
        df = df[['query_id', 'query', 'passage', 'alternatives', 'answer', 'answer_pred', 'r_flag']]
        csv_path = os.path.join('result', 'emsemble'+'_val.csv')
        df.to_csv(csv_path, index=False)

    print('time:%d' % (time.time()-time0))


if __name__ == '__main__':
    is_ensemble = True
    if is_ensemble:
        time0 = time.time()
        print('ensemble...')
        config = config_ensemble.config
        is_true_test = config.is_true_test
        if is_true_test:
            flag = '_submission.pkl'
        else:
            flag = '_val.pkl'
        model_lst = config.model_lst
        print('model num:%d' % len(model_lst))

        config_lst = [config_m_reader.config, config_m_reader_plus.config]

        model_name = [
            ['m_reader_1', 'm_reader_2', 'm_reader_3', 'm_reader_4', 'm_reader_5', 'm_reader_6'],
            ['m_reader_plus_1']
        ]

        print('start single model...')
        for i in range(len(config_lst)):
            cfg = config_lst[i]
            config = cfg
            mdl_lst = model_name[i]
            for mdl in mdl_lst:
                config.is_true_test = is_true_test
                config.test_batch_size = 64
                config.model_test = mdl
                if (mdl+flag) in model_lst and os.path.isfile(os.path.join('result/ans_range', mdl+flag)) is False:
                    print('gen ', os.path.join('result/ans_range', mdl+flag))
                    test(config)
                elif (mdl+flag) in model_lst:
                    print(os.path.join('result/ans_range', mdl+flag), ', exiting')
        config = config_ensemble.config
        test_ensemble(config)
        print('ensemble time:%d' % (time.time()-time0))
    else:
        print('single model....')
        config = config_m_reader.config
        test(config)


















