# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'stanford_ar'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 12
    mode = 'LSTM'
    batch_size = 64
    hidden_size = 256
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100
    val_mean = True  # 这个指标用来衡量，是否是每隔固定次数验证一次

    # 测试
    model_test = 'stanford_ar_1'
    is_true_test = False

config = Config()
