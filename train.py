# coding = utf-8
# author = xy

import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import loader
import torch
from torch import optim
from torch import nn
import preprocess_data
import utils
from modules.layers import loss
import build_dataset

from config import config_base
from config import config_m_reader

from modules import m_reader

config = config_m_reader.config


def train():
    time_start = time.time()

    # prepare
    preprocess_data.gen_pre_file_for_train()

    # load w2v
    embedding_np = loader.load_w2v(config.train_embedding + '.npy')

    # prepare: train_df
    preprocess_data.gen_train_val_datafile()

    # load data
    print('load data...')
    time0 = time.time()
    # load train data
    if os.path.isfile(config.train_pkl):
        with open(config.train_pkl, 'rb') as file:
            train_data = pickle.load(file)
    else:
        train_data = build_dataset.CustomDataset(
            df_file=config.train_df,
            vocab_path=config.train_vocab_path,
            tag_path=config.tag_path
        )
        with open(config.train_pkl, 'wb') as file:
            pickle.dump(train_data, file)

    # load val data
    if os.path.isfile(config.val_pkl):
        with open(config.val_pkl, 'rb') as file:
            val_data = pickle.load(file)
    else:
        val_data = build_dataset.CustomDataset(
            df_file=config.val_df,
            vocab_path=config.train_vocab_path,
            tag_path=config.tag_path
        )
        with open(config.val_pkl, 'wb') as file:
            pickle.dump(val_data, file)
    print('train data size:%d, val data size:%d, time:%d' % (train_data.__len__(), val_data.__len__(), time.time()-time0))

    # build train, val dataloader
    train_loader = loader.build_loader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = loader.build_loader(
        dataset=val_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    # model:
    param = {
        'embedding': embedding_np,
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
    # 改变embedding_fix
    model.embedding.sd_embedding.embedding_fix = nn.Embedding(
        num_embeddings=embedding_np.shape[0],
        embedding_dim=embedding_np.shape[1],
        padding_idx=0,
        _weight=torch.Tensor(embedding_np)
    )
    model.embedding.sd_embedding.embedding_fix.weight.requires_grad = False
    model = model.cuda()

    # loss
    criterion = loss.LossJoin()

    # optimizer
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(optimizer_param, lr=config.lr, weight_decay=1e-5)

    # load model param, optimizer param, train param
    model_path = os.path.join('model', config.model_save)
    if os.path.isfile(model_path):
        print('load training param, ', model_path)
        state = torch.load(model_path)
        model.load_state_dict(state['cur_model_state'])
        optimizer.load_state_dict(state['cur_opt_state'])
        epoch_list = range(state['cur_epoch']+1, state['cur_epoch']+1+config.epoch)
        train_loss_list = state['train_loss']
        val_loss_list = state['val_loss']
        val_accuracy = state['val_accuracy']
        steps = state['steps']
        time_use = state['time']
    else:
        state = None
        epoch_list = range(config.epoch)
        train_loss_list = []
        val_loss_list = []
        val_accuracy = []
        steps = []
        time_use = 0

    # train
    model_param_num = 0
    for param in model.parameters():
        if param.requires_grad is True:
            model_param_num += param.nelement()

    print('starting training: %s' % config.model_name)
    if state is None:
        print('start_epoch:0, end_epoch:%d, num_params:%d' %
              (config.epoch-1, model_param_num))
    else:
        print('start_epoch:%d, end_epoch:%d, num_params:%d' %
              (state['cur_epoch']+1, state['cur_epoch']+config.epoch, model_param_num))

    plt.ion()
    train_loss = 0
    train_c = 0
    flag = False
    cc = 0
    grade_1 = False
    grade_2 = False

    for e in epoch_list:
        for i, batch in enumerate(train_loader):
            # cuda
            batch = utils.deal_batch(batch)
            model.train()
            optimizer.zero_grad()
            outputs = model(batch)
            loss_value = criterion(outputs, batch[-1].view(-1))
            loss_value.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad)
            optimizer.step()

            train_loss += loss_value.item()
            train_c += 1

            if config.val_mean:
                flag = (train_c % config.val_every == 0)
            else:
                if (train_c % (config.val_every//2) == 0) and (cc <= 0):
                    cc += 1
                    flag = True
                elif grade_1 and (train_c % (config.val_every*78) == 0):
                    flag = True
                elif grade_2 and (train_c % config.val_every == 0):
                    flag = True

            if flag:
                flag = False
                val_loss = 0
                val_c = 0
                correct_num = 0
                correct_01_num = 0
                correct_2_num = 0
                sum_num = 0
                sum_01_num = 0
                sum_2_num = 0
                with torch.no_grad():
                    model.eval()
                    for val_batch in val_loader:
                        # cut, cuda
                        val_batch = utils.deal_batch(val_batch)
                        outputs = model(val_batch)

                        loss_value = criterion(outputs, val_batch[-1].view(-1))
                        _, k = torch.max(outputs, dim=1)

                        k = k.view(-1)
                        correct_num += torch.sum(k == val_batch[-1].view(-1)).item()
                        sum_num += val_batch[-1].size(0)

                        mask_01 = val_batch[-1].view(-1).eq(2)
                        p_01 = val_batch[-1].view(-1).masked_fill(mask_01, 4)
                        correct_01_num += torch.sum(k == p_01).item()
                        sum_01_num += torch.sum(val_batch[-1].view(-1).ne(2)).item()

                        mask_2 = val_batch[-1].view(-1).ne(2)
                        p_02 = val_batch[-1].view(-1).masked_fill(mask_2, 5)
                        correct_2_num += torch.sum(k == p_02).item()
                        sum_2_num += torch.sum(val_batch[-1].view(-1).eq(2)).item()

                        val_loss += loss_value.item()
                        val_c += 1

                train_loss_list.append(train_loss/train_c)
                val_loss_list.append(val_loss/val_c)
                steps.append(train_c)
                val_accuracy.append(correct_num*1.0/sum_num)

                print('training, epochs:%2d, steps:%5d, train_loss:%.4f, val_loss:%.4f, val_accuracy:%.4f, '
                      'val_accuracy01:%.4f, val_accuracy2:%.4f, time:%4ds' %
                      (e, sum(steps), train_loss/train_c, val_loss/val_c, correct_num*1.0/sum_num,
                       correct_01_num*1.0/sum_01_num, correct_2_num*1.0/sum_2_num, time.time()-time_start+time_use))

                if val_loss/val_c > 0.65:
                    grade_1 = True
                    grade_2 = False
                else:
                    grade_1 = False
                    grade_2 = True

                train_loss = 0
                train_c = 0

                # draw
                plt.cla()
                x = np.cumsum(steps)
                plt.plot(
                    x,
                    train_loss_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_list,
                    color='b',
                    label='val'
                )
                # plt.plot(
                #     x,
                #     val_accuracy,
                #     color='black',
                #     label='accuracy'
                # )

                plt.xlabel('steps')
                plt.ylabel('loss/accuracy')
                plt.legend()
                plt.pause(0.0000001)

                fig_path = os.path.join('model', config.model_save+'.png')
                plt.savefig(fig_path)
                plt.show()

                # save model
                if os.path.isfile(model_path):
                    state = torch.load(model_path)
                else:
                    state = {}

                if state == {} or state['best_loss'] > (val_loss/val_c):
                    state['best_model_state'] = model.state_dict()
                    state['best_opt_state'] = optimizer.state_dict()
                    state['best_loss'] = val_loss/val_c
                    state['best_val_accuracy'] = correct_num*1.0/sum_num
                    state['best_epoch'] = e
                    state['best_step'] = sum(steps)
                    state['best_time'] = time_use + time.time() - time_start

                state['cur_model_state'] = model.state_dict()
                state['cur_opt_state'] = optimizer.state_dict()
                state['cur_epoch'] = e
                state['train_loss'] = train_loss_list
                state['val_loss'] = val_loss_list
                state['val_accuracy'] = val_accuracy
                state['steps'] = steps
                state['time'] = time_use + time.time() - time_start

                torch.save(state, model_path)


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    train()
