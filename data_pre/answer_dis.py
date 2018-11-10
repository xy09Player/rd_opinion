# coding = utf-8
# author = xy

from preprocess_data import organize_data
from config import config_base
import pickle


# 获得答案分布
def get_ans_dis():
    data = '../data/train/train.json'
    df = organize_data(data)

    answers = df['answer'].values
    answers = [a.strip() for a in answers]

    answer_dict = {}
    for answer in answers:
        if answer in answer_dict:
            answer_dict[answer] += 1
        else:
            answer_dict[answer] = 1

    answer_dict_tmp = {}
    for k, v in answer_dict.items():
        answer_dict_tmp[k] = v/len(df)
    answer_dict = answer_dict_tmp

    file_out = '../data_gen/ans_dis.pkl'
    with open(file_out, 'wb') as file:
        pickle.dump(answer_dict, file)


if __name__ == '__main__':
    get_ans_dis()
