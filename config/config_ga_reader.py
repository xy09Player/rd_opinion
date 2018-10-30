# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'ga_reader'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 12
    mode = 'GRU'
    batch_size = 32
    hidden_size = 128
    encoder_layer_num = 1
    dropout_p = 0.3
    val_every = 100
    val_mean = False  # 这个指标用来衡量，是否是每隔固定次数验证一次

    # 测试
    model_test = 'ga_reader_1'
    is_true_test = False

config = Config()
