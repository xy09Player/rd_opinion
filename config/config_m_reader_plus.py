# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'm_reader_plus'
    model_save = model_name + '_1'
    lr = 1e-4
    weight_decay = 0
    is_bn = True
    epoch = 10
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 150
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100
    val_mean = False  # 这个指标用来衡量，是否是每隔固定次数验证一次

    # 测试
    model_test = 'm_reader_plus_2'
    is_true_test = False

config = Config()
