# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    bi_daf_1 = 'bi_daf_1'  # 0.7292
    bi_daf_2 = 'bi_daf_2'  # 0.7303
    bi_daf_3 = 'bi_daf_3'  # 0.7319
    bi_daf_4 = 'bi_daf_4'  # 0.7321

    m_reader_1 = 'm_reader_1'  # 0.7371
    m_reader_2 = 'm_reader_2'  # 0.7383
    m_reader_3 = 'm_reader_3'  # 0.734
    m_reader_4 = 'm_reader_4'  # 0.7334
    m_reader_5 = 'm_reader_5'  # 0.7349

    m_reader_plus_1 = 'm_reader_plus_1'  # 0.733
    m_reader_plus_2 = 'm_reader_plus_2'  # 0.7281

    model_lst = [bi_daf_4, m_reader_2, m_reader_plus_1]
    model_weight = [0.7321, 0.7383, 0.733]

    # model_weight = utils.softmax(model_weight)
    model_weight = utils.mean(model_weight)

    is_true_test = True

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
