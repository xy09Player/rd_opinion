# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/train/train.json'
    train_data_1 = 'data/train/train_baidu.json'
    train_data_2 = 'data/train/train_xiaoniu.json'
    val_data = 'data/val/val.json'
    test_data = 'data/test/test.json'

    train_df = 'data_gen/train_df.csv'
    val_df = 'data_gen/val_df.csv'
    test_df = 'data_gen/test_df.csv'

    train_pkl = 'data_gen/train_df.pkl'
    val_pkl = 'data_gen/val_df.pkl'
    val_true_pkl = 'data_gen/val_true_df.pkl'
    test_pkl = 'data_gen/test_df.pkl'

    train_vocab_path = 'data_gen/train_vocab.pkl'
    test_vocab_path = 'data_gen/test_vocab.pkl'
    tag_path = 'data_gen/tag2index.pkl'

    pre_embedding_zh = 'data/merge_sgns_bigram_char300.txt'
    pre_embedding_en = 'data/glove300.txt'
    train_embedding = 'data_gen/train_embedding'
    test_embedding = 'data_gen/test_embedding'

    submission = 'submission/result.txt'

    shorten_sentence_num = 2

    batch_size = 32
    test_batch_size = 64
    mode = 'LSTM'
    hidden_size = 150
    dropout_p = 0.2
    encoder_dropout_p = 0.1
    encoder_bidirectional = True
    encoder_layer_num = 1
    is_bn = True
    lr = 1e-4
    max_grad = 10

    val_print_ratio = 5
    val_split_value = 0.67

    num_align_hops = 2

    k = 2




config = ConfigBase()
