# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'dfn_net'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 10
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 100
    encoder_layer_num = 1
    dropout_p = 0.1
    val_every = 100
    val_mean = True  # 这个指标用来衡量，是否是每隔固定次数验证一次
    val_split_value = 0.5
    all_val_data = True  # 是否使用所有验证数据进行验证

    k = 2  # 答案评分层迭代次数

    # 测试
    model_test = 'dfn_net_1'
    is_true_test = True

config = Config()
