# coding = utf-8
# author = xy

from data_pre import trans_helper
from data_pre import clean_data
import json
import time
from multiprocessing import Pool
import numpy as np


def split_data(json_in, start=True, shuffle=True, num=6):
    if start:
        data = []
        with open(json_in, 'r', encoding='utf-8') as file:
            for sentence in file.readlines():
                d = json.loads(sentence)
                data.append(d)
    else:
        with open(json_in, 'r', encoding='utf-8') as file:
            data = json.load(file)

    if shuffle:
        np.random.seed(33)
        np.random.shuffle(data)

    chuck_num = int(len(data) / num)
    for i in range(num):
        if i < num-1:
            data_tmp = data[i*chuck_num: chuck_num*(i+1)]
        else:
            data_tmp = data[i*chuck_num:]

        with open(json_in[: -5] + '_' + str(i) + '.json', 'w') as file:
            json.dump(data_tmp, file)


def merge_data(json_ins, json_out):
    data = []
    for file_path in json_ins:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_tmp = json.load(file)
            data = data + data_tmp

    c_num = 0
    for d in data:
        if d['flag'] is False:
            c_num += 1
    print('need deal data_num:%d/%d' % (c_num, len(data)))

    with open(json_out, 'w') as file:
        json.dump(data, file)


def deal(d, is_xiaoniu, is_youdao, is_baidu, is_bai_pachong, apikey, secretKey):
    if d['flag']:
        return d

    passage = d['passage'].strip()

    # 小牛翻译
    if is_xiaoniu:
        passage_en, flag_en = trans_helper.xiaoniu_translate(passage, 'zh', 'en', apikey)
        passage_zh, flag_zh = trans_helper.xiaoniu_translate(passage_en, 'en', 'zh', apikey)

        if flag_en + flag_zh != 0:
            passage_list = passage.split('。')
            result = []
            flag = True
            for p in passage_list:
                if p.strip() == '':
                    continue
                p_en, p_flag_en = trans_helper.xiaoniu_translate(p, 'zh', 'en', apikey)
                p_zh, p_flag_zh = trans_helper.xiaoniu_translate(p_en, 'en', 'zh', apikey)
                result.append(p_zh)
                if p_flag_zh + p_flag_en != 0:
                    flag = False
            result = '。'.join(result)

            if flag:
                d['flag'] = flag
                d['passage'] = result


    # # 有道翻译
    # if d['is_youdao'] == '1':
    #     passage_en, flag_en = trans_helper.youdao_translate(passage, 'zh', 'en', d['apikey'], d['secretKey'])
    #     passage_zh, flag_zh = trans_helper.youdao_translate(passage_en, 'en', 'zh', d['apikey'], d['secretKey'])
    #

    # 百度翻译
    if is_baidu:
        passage_en, flag_en = trans_helper.baidu_translate(passage, 'zh', 'en', apikey, secretKey)
        passage_zh, flag_zh = trans_helper.baidu_translate(passage_en, 'en', 'zh', apikey, secretKey)

        if flag_en + flag_zh != 0:
            passage_list = passage.split('。')
            result = []
            flag = True
            for p in passage_list:
                if p.strip() == '':
                    continue
                p_en, p_flag_en = trans_helper.baidu_translate(p, 'zh', 'en', apikey, secretKey)
                p_zh, p_flag_zh = trans_helper.baidu_translate(p_en, 'en', 'zh', apikey, secretKey)
                result.append(p_zh)
                if p_flag_zh + p_flag_en != 0:
                    flag = False
            result = '。'.join(result)

            if flag:
                d['flag'] = flag
                d['passage'] = result

    if flag_en + flag_zh == 0:
        d['flag'] = True
        d['passage'] = passage_zh

    return d


def trans(json_in, json_out, start=True, shuffle=True):
    if start:
        data = []
        with open(json_in, 'r', encoding='utf-8') as file:
            for sentence in file.readlines():
                d = json.loads(sentence)
                d['flag'] = False
                data.append(d)
    else:
        with open(json_in, 'r', encoding='utf-8') as file:
            data = json.load(file)

    # 随机扰乱顺序
    if shuffle:
        np.random.seed(33)
        np.random.shuffle(data)

    # 预处理
    for i in range(len(data)):
        d = data[i]
        passage = d['passage']
        passage = clean_data.deal_data([passage])[0]
        d['passage'] = passage
        data[i] = d

    c_num = 0
    for d in data:
        if d['flag'] is False:
            c_num += 1
    print('need deal data_num:%d/%d' % (c_num, len(data)))

    # 百度翻译
    # baidu_apikeys = [# '20181104000229891',  # 曾维新
    #                  '20181104000229916',  # bb
    #                  '20181105000230263',  # bb2
    #                  '20181105000230199',  # 志飞
    #                  '20181105000230206',  # 大海
    #                  '20181105000230223',  # 老谭
    #                  '20181105000230246',  # 小号1
    #                  '20181105000230254',  # 老谭2
    #                  '20181105000230256',  # 老谭3
    #                  '20181105000230258',  # 庞宁
    #                  '20181105000230261',  # 李祯
    #                  '20181105000230119',  # 谭真
    #                  '20181106000230588',  # 妈
    #                  '20181106000230594',  # 俊峰
    #                  '20181106000230790',  # 瓴宇
    #                  '20181106000230804',  # 大娘
    #                  ]
    # baidu_secretKey = [# 'Z3kcAfpfFuwqvwN3Ixfd',
    #                    '_DQpFHMJVgYclCEkfNMZ',
    #                    'kEC7mHa8lxNgYdbjiRQT',
    #                    'EiPvDKwa3IGKmtSgrkqX',
    #                    'sBAvNIyyVCz2Cy29YZX9',
    #                    'O2N1KikFl0vg1HkSrKLy',
    #                    'QWD_DGSJ4FoKahRquNXU',
    #                    'Ja489s4QQxrSaTbZPuBP',
    #                    '7ddAECYZN5tLU8tCH9cp',
    #                    'bTXuXjnJaIfWVeNdHHif',
    #                    'Iz6bGbq3mCsJV29CwRSJ',
    #                    'n8O3SUBXTqAMpx4HNC9d',
    #                    'gEbeY6UCYdF8ldFKE6wY',
    #                    'BhhE3YU6bsaK3lPNkgOP',
    #                    'aI_IwPbWdCY3o74z4KyF',
    #                    'NSxha5lN7E6vzn0Zz48w'
    #                    ]
    # time0 = time.time()
    # num = 0
    # err_num = 0
    # index = 0
    # apikey = baidu_apikeys[index]
    # secretKey = baidu_secretKey[index]
    # for i in range(len(data)):
    #     d = deal(data[i], is_xiaoniu=False, is_youdao=False, is_baidu=True, is_bai_pachong=False, apikey=apikey, secretKey=secretKey)
    #     if d['flag'] is False:
    #         err_num += 1
    #     else:
    #         err_num = 0
    #         data[i] = d
    #     num += 1
    #
    #     if err_num % 10 == 0 and err_num != 0:
    #         index += 1
    #         if index < len(baidu_apikeys):
    #             apikey = baidu_apikeys[index]
    #             secretKey = baidu_secretKey[index]
    #             err_num = 0
    #         else:
    #             cc_num = 0
    #             for dd in data:
    #                 if dd['flag'] is False:
    #                     cc_num += 1
    #             print('final, num:%d, deal_data_num:%d/%d, use time:%d' % (num, cc_num, len(data), time.time()-time0))
    #
    #             with open(json_out, 'w') as file:
    #                 json.dump(data, file)
    #             exit(0)
    #
    #     if num % 2500 == 0 or num == 10:
    #         cc_num = 0
    #         for dd in data:
    #             if dd['flag'] is False:
    #                 cc_num += 1
    #         print('num:%d, deal_data_num:%d/%d, use time:%d' % (num, cc_num, len(data), time.time()-time0))
    #
    #         with open(json_out, 'w') as file:
    #             json.dump(data, file)

    # 小牛翻译
    xiaoniu_apikeys = [#'42403ddf41072fd9771dddc13acc0900',  # bb
                       #'a40f0b3e5b40bdffa335b7d720cbd45f',  # bb
                       # '5a979c2ec7a4756ede0cf63312f2a1b4',  # 卢聪
                       # '288730da6848f9d4efa97c5fc64faf7a',  # cc
                       # '67ae095d4ea80e2d3c8bc1ad479489e0',  # 老谭1
                       # '940f33e891a6f6a0f2d9eb8c0da55b22',  # 老谭2
                       # '99bbc6e59a9af7c4dda1c767feb62406',  # 杨伟龙
                       # 'f9f447aa36f8064092e717ad9d9b21b2',  # 梅阳
                       # 'cecab0774a12583effc646cf95376972',  # 瓴宇
                       # 'd4287120b398280f80ce638fd765480d',  # 真博
                       # 'eda25af81558904bd9d64034ecdc0924',  # 宇恒
                       # 'e1e1bc58768e1355ec2d65dcdbb268ca',  # 猴哥
                       # '9493999482c68ef9a535d0a3d9eb02dc',  # 劲智
                       # '960f0b40eca97a46ad594f24e7c4a555',  # 志飞
                       # '517f50719f1b242e633538ff99f356ea',  # 贺云岳
                       # '3de74e63611b05fb61308e58fbe43d6d',  # 张鹏乐
                       # '75f1bcccca8590e936aca68c8c5055df',  # 脸盆
                       # '4fbafe869644bc480f58dcecf8c08a9e',  # 瓴宇2
                       # 'd74813d54332b5278d3cf734fbdb0043',  # 瓴宇3
                       # '04b27ccb1f37fc188260b48eb049e28c',   # xy2
                       # '57f5f10c107ad049ab73e1526a282d75',   # 妈
                       #  'd30d16be1a556f63c95af043d889c325',  # 俊峰
                       #  'c8327943b6c21c91d20e7a9613ae3152',  # 小乐
                       #  'ebf915efaa874090be4da13faba34883',  # 地雷
                        'a78bf9cbdbd4d2d1c5b152382d963a7c',  # 赛斌

                       ]
    time0 = time.time()
    num = 0
    err_num = 0
    index = 0
    apikey = xiaoniu_apikeys[index]
    secretKey = None
    for i in range(len(data)):
        d = deal(data[i], is_xiaoniu=True, is_youdao=False, is_baidu=False, is_bai_pachong=False, apikey=apikey, secretKey=secretKey)
        if d['flag'] is False:
            err_num += 1
        else:
            err_num = 0
            data[i] = d
        num += 1

        if err_num % 20 == 0 and err_num != 0:
            index += 1
            if index < len(xiaoniu_apikeys):
                apikey = xiaoniu_apikeys[index]
                err_num = 0
            else:
                cc_num = 0
                for dd in data:
                    if dd['flag'] is False:
                        cc_num += 1
                print('final, num:%d, deal_data_num:%d/%d, use time:%d, index:%d' % (num, cc_num, len(data), time.time()-time0, index))

                with open(json_out, 'w') as file:
                    json.dump(data, file)
                exit(0)

        if num % 2500 == 0 or num == 10:
            cc_num = 0
            for dd in data:
                if dd['flag'] is False:
                    cc_num += 1
            print('num:%d, deal_data_num:%d/%d, use time:%d, index:%d' % (num, cc_num, len(data), time.time()-time0, index))

            with open(json_out, 'w') as file:
                json.dump(data, file)

    with open(json_out, 'w') as file:
        json.dump(data, file)

    # # 有道翻译
    # youdao_apikeys = ['186483e9dd22f401',  # xy
    #                   ]
    # youdao_secretKey = ['Y1mX21LrPIVR7VT41wBQLz6BqyPfd4Mu']
    # for i in range(len(youdao_apikeys)):
    #     time0 = time.time()
    #     data_tmp = []
    #     for d in data:
    #         d_tmp = {
    #             'd': d,
    #             'is_xiaoniu': '0',
    #             'is_youdao': '1',
    #             'is_baidu': '0',
    #             'is_bai_pachong': '0',
    #             'apikey': youdao_apikeys[i],
    #             'secretKey': youdao_secretKey[i]
    #         }
    #         data_tmp.append(d_tmp)
    #     data = data_tmp
    #     with Pool(processes=agents) as pool:
    #         result = pool.map(deal, data)
    #     data = result
    #
    #     err_num = 0
    #     for d in data:
    #         if d['flag'] is False:
    #             err_num += 1
    #
    #     with open(json_out, 'w') as file:
    #         json.dump(data, file)
    #
    #     print('youdao fanyi, time:%d, err_num:%d/%d' % (time.time()-time0, err_num, len(data)))



if __name__ == '__main__':
    # 切分数据
    if False:
        json_in = '../data/train/train_merge_ddd.json'
        split_data(json_in, start=False, shuffle=True, num=10)

    # 翻译
    if False:
        json_in = '../data/train/train_ggg_split/train_merge_ggg.json'
        json_out = '../data/train/train_ggg_split/train_xiaoniu.json'
        print(json_in)
        trans(json_in, json_out, start=False)

    # 合并数据
    if False:
        json_ins = ['../data/train/baidu_fanyi/train_f_0.json',
                    '../data/train/baidu_fanyi/train_f_1.json',
                    '../data/train/baidu_fanyi/train_f_2.json',
                    '../data/train/baidu_fanyi/train_f_3.json',
                    '../data/train/baidu_fanyi/train_f_4.json',
                    '../data/train/baidu_fanyi/train_f_5.json',
                    '../data/train/baidu_fanyi/train_f_6.json',
                    '../data/train/baidu_fanyi/train_f_7.json',
                    '../data/train/baidu_fanyi/train_f_8.json',
                    '../data/train/baidu_fanyi/train_f_9.json'
                    ]
        json_out = '../data/train/baidu_fanyi/train_baidu.json'
        merge_data(json_ins, json_out)
























