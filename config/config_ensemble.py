# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    m_reader_1 = 'm_reader_1'  # 0.7507
    m_reader_2 = 'm_reader_2'  # 0.7531
    m_reader_3 = 'm_reader_3'  # 0.7514
    m_reader_4 = 'm_reader_4'  # 0.7525
    m_reader_5 = 'm_reader_5'  # 0.756
    m_reader_6 = 'm_reader_6'  # 0.752
    m_reader_9 = 'm_reader_9'  # 0.755

    model_lst = [m_reader_2, m_reader_3, m_reader_4, m_reader_5, m_reader_6]
    model_weight = [0.7531, 0.7514, 0.7525, 0.756, 0.752]

    # model_lst = [m_reader_2, m_reader_5, m_reader_9]
    # model_weight = [0.753, 0.756, 0.755]

    # model_weight = utils.softmax(model_weight)
    model_weight = utils.mean(model_weight)

    is_true_test = True

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
