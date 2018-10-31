# encoding = utf-8
# author = xy

import torch
import jieba
from jieba import posseg
import numpy as np
import pickle


def pad(data_array, length):
    """ padding """
    tmp = []
    for d in data_array:
        if len(d) > length:
            tmp.append(d[: length])
        elif len(d) < length:
            tmp.append(d + [0]*(length-len(d)))
        else:
            tmp.append(d)
    data_array = tmp
    return data_array


def deal_batch(batch):
    """
    deal batch: cuda
    :param batch:[content_index, content_flag, content_is_in_title, content_is_in_question, question_index,
    question_flag, start, end] or [content_index, content_flag, content_is_in_title, content_is_in_question,
    question_index, question_flag]
    :return: batch_done
    """
    batch = [b.cuda() for b in batch]

    return batch


def get_mask(tensor, padding_idx=0):
    """ get mask tensor """
    return torch.ne(tensor, padding_idx).float()


def masked_flip(seq_tensor, mask):
    """
     flip seq_tensor
    :param seq_tensor: (seq_len, batch_size, input_size)
    :param mask: (batch_size, seq_len)
    :return: (seq_len, batch_size, input_size)
    """
    length = mask.eq(1).long().sum(dim=1)
    batch_size = seq_tensor.size(1)

    outputs = []
    for i in range(batch_size):
        temp = seq_tensor[:, i, :]
        temp_length = length[i]

        idx = list(range(temp_length-1, -1, -1)) + list(range(temp_length, seq_tensor.size(0)))
        idx = seq_tensor.new_tensor(idx, dtype=torch.long)

        temp = temp.index_select(0, idx)
        outputs.append(temp)

    outputs = torch.stack(outputs, dim=1)
    return outputs


def softmax(weight):
    exp = np.exp(weight)
    return exp / exp.sum()


def mean(weight):
    weight = np.array(weight)
    weight = weight / sum(weight)
    return weight


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


def words2index(words_list, vocab_path):
    """
    :param words_list: list of list
    :param vocab_path: file_path
    :return: list of list
    """
    with open(vocab_path, 'rb') as file:
        lang = pickle.load(file)
        w2i = lang['w2i']

    result = []
    for words in words_list:
        tmp = [w2i[word] if word in w2i else w2i['<unk>'] for word in words]
        result.append(tmp)

    return result


def tags2index(tags_list, tag_path):
    """
    :param tags_list:  list of list
    :param tag_path: file_path
    :return: list of list
    """
    with open(tag_path, 'rb') as file:
        lang = pickle.load(file)

    result = []
    for tags in tags_list:
        tmp = [lang[tag] if tag in lang else lang['<unk>']for tag in tags]
        result.append(tmp)

    return result


def split_word(s, have_tag=False):
    """
    分词
    :param s: str
    :return: list
    """
    # jieba
    if have_tag is False:
        word_list = jieba.lcut(s, HMM=False)
        return word_list
    else:
        word_list, tag_list = list(zip(*posseg.lcut(s, HMM=False)))
        return word_list, tag_list


def compute_mean(vec, mask):
    """
    :param vec: (c_len, batch_size, input_size)
    :param mask: (batch_size, c_len)
    :return: (batch_size, input_size)
    """
    result = []
    vec = vec.transpose(0, 1)
    batch_size = vec.size(0)
    mask = mask.long().sum(1)
    for i in range(batch_size):
        vec_i = vec[i][: mask[i].item()]
        vec_i = torch.mean(vec_i, dim=0)
        result.append(vec_i)
    result = torch.stack(result)
    return result


def get_last_state(h, h_mask):
    """
    :param h: (batch_size, h_len, hidden_size)
    :param h_mask: (batch_size, h_len)
    :return: last_state: (batch_size, hidden_size)
    """
    lens = h_mask.sum(1).long()
    lens = (lens-1).view(-1, 1).expand(-1, h.size()[-1]).unsqueeze(1)  # (batch_size, 1, hidden_size)
    last_state = h.gather(1, lens).squeeze(1)  # (batch_size, hidden_size)
    return last_state

