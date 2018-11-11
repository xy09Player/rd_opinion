# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    m_reader_1 = 'm_reader_1'  # 0.7506
    m_reader_2 = 'm_reader_2'  # 0.753
    m_reader_3 = 'm_reader_3'  # 0.7513
    m_reader_4 = 'm_reader_4'  # 0.7524
    m_reader_5 = 'm_reader_5'  # 0.7558
    m_reader_6 = 'm_reader_6'  # 0.7519

    model_lst = [m_reader_2, m_reader_3, m_reader_4, m_reader_5, m_reader_6]
    model_weight = [0.753, 0.7513, 0.7524, 0.7558, 0.7519]

    # model_weight = utils.softmax(model_weight)
    model_weight = utils.mean(model_weight)

    is_true_test = False

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
