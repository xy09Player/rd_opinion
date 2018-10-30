# coding = utf-8
# author = xy

import os
import time
import pickle
import pandas as pd
import torch
from torch import nn
from data_pre import wfqd
import loader
import numpy as np
import utils
import preprocess_data
from config import config_base
from config import config_m_reader
from config import config_ensemble

from modules import m_reader


config = config_m_reader.config
# config = config_ensemble.config


def test():
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
        if os.path.isfile(config.val_pkl):
            with open(config.val_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.val_df, config.train_vocab_path, config.tag_path)
            with open(config.val_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    else:
        if os.path.isfile(config.test_pkl):
            with open(config.test_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.test_df, config.test_vocab_path, config.tag_path)
            with open(config.test_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data[:9],
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
        df = pd.read_csv(config.test_df)
    else:
        df = pd.read_csv(config.val_df)

    # 生成结果
    zhenglis = df['zhengli'].values
    fulis = df['fuli'].values
    alts = df['alternatives'].values
    wfqd_list = wfqd.wfqd_list
    tmp = []
    for r, z, f, alt in zip(result, zhenglis, fulis, alts):
        alt_list = alt.split('|')
        if r == 0:
            if z == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif z == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif z == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==0, meet wrong data')
        elif r == 1:
            if f == alt_list[0].strip():
                tmp.append(alt_list[0])
            elif f == alt_list[1].strip():
                tmp.append(alt_list[1])
            elif f == alt_list[2].strip():
                tmp.append(alt_list[2])
            else:
                print('r==1, meet wrong data')
        else:
            if alt_list[0].strip() in wfqd_list:
                tmp.append(alt_list[0])
            elif alt_list[1].strip() in wfqd_list:
                tmp.append(alt_list[1])
            elif alt_list[2].strip() in wfqd_list:
                tmp.append(alt_list[2])
            else:
                print('r==2, meet wfqd of not in wfqd')
                tmp.append(alt_list[-1])

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


def test_ensemble():
    time0 = time.time()

    if config.is_true_test:
        df = pd.read_csv(config.test_df)
    else:
        df = pd.read_csv(config.val_df)

    # 加权求和
    model_lst = config.model_lst
    model_weight = config.model_weight
    range_ensemble = np.zeros([len(df), 3])
    for ml, mw in zip(model_lst, model_weight):
        result_path = os.path.join('result/ans_range', ml)
        ans_range = torch.load(result_path)
        ans_range = np.array(ans_range)
        range_ensemble += ans_range * mw

    result = np.argmax(range_ensemble, axis=1)
    result = result.tolist()

    # 生成结果
    alts = df['alternatives'].values
    assert len(alts) == len(result)
    tmp = []
    for a, r in zip(alts, result):
        # trick
        if False:
            print('trick')
            continue

        a_list = a.split('|')
        a = a_list[r]
        tmp.append(a)

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
        csv_path = os.path.join('result', 'val(ensemble).csv')
        df.to_csv(csv_path, index=False)

    print('time:%d' % (time.time()-time0))


if __name__ == '__main__':
    if config == config_ensemble.config:
        print('ensemble...')
        test_ensemble()
    else:
        print('single model...')
        test()
