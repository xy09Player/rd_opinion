# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'bi_daf'
    model_save = model_name + '_4'
    is_bn = True
    epoch = 10
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 128
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100
    val_mean = False  # 这个指标用来衡量，是否是每隔固定次数验证一次
    all_val_data = True  # 是否使用所有验证数据进行验证

    # 测试
    model_test = 'bi_daf_4'
    is_true_test = True

config = Config()
