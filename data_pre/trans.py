# coding = utf-8
# author = xy

from data_pre import trans_helper
from data_pre import clean_data
import json
import time
from multiprocessing import Pool


def split_data(json_in):
    data = []
    c_num = 0
    with open(json_in, 'r', encoding='utf-8') as file:
        for sentence in file.readlines():
            d = json.loads(sentence)
            c_num += 1
            data.append(d)

    chuck_num = int(len(data) / 10)
    for i in range(10):
        if i < 9:
            data_tmp = data[i*chuck_num: chuck_num*(i+1)]
        else:
            data_tmp = data[i*chuck_num:]

        with open(json_in[: -5] + '_' + str(i) + '.json', 'w') as file:
            json.dump(data_tmp, file)


def deal(d):
    # is_xiaoniu=False, is_youdao=False, is_baidu=False, is_bai_pachong=False, apikey=None, secretKey=None
    if d['d']['flag']:
        return d['d']

    passage = d['d']['passage']
    passage = clean_data.deal_data([passage])[0]

    # 小牛翻译
    if d['is_xiaoniu'] == '1':
        passage_en, flag_en = trans_helper.xiaoniu_translate(passage, 'zh', 'en', d['apikey'])
        passage_zh, flag_zh = trans_helper.xiaoniu_translate(passage_en, 'en', 'zh', d['apikey'])

    # 有道翻译
    if d['is_youdao'] == '1':
        passage_en, flag_en = trans_helper.youdao_translate(passage, 'zh', 'en', d['apikey'], d['secretKey'])
        passage_zh, flag_zh = trans_helper.youdao_translate(passage_en, 'en', 'zh', d['apikey'], d['secretKey'])

    if flag_en + flag_zh == 0:
        d['d']['flag'] = True
        d['d']['passage'] = passage_zh

    return d['d']



    # baidu
    # passage_en = trans_helper.baidu_translate(passage, 'zh', 'en')
    # passage_zh = trans_helper.baidu_translate(passage_en, 'en', 'zh')

    # 小牛
    # passage_en, flag_en = trans_helper.xiaoniu_translate(passage, 'zh', 'en')
    # passage_zh, flag_zh = trans_helper.xiaoniu_translate(passage_en, 'en', 'zh')

    # 有道
    # passage_en, flag_en = trans_helper.youdao_translate(passage, 'zh', 'en')
    # passage_zh, flag_zh = trans_helper.youdao_translate(passage_en, 'en', 'zh')





def trans(json_in, json_out):
    c_num = 0
    data = []
    with open(json_in, 'r', encoding='utf-8') as file:
        for sentence in file.readlines():
            d = json.loads(sentence)
            d['flag'] = False
            c_num += 1
            data.append(d)

    agents = 8

    # # 百度翻译
    # baidu_apikeys = ['20181104000229891',  # 曾维新
    #                  '20181104000229916',  # bb
    #                  ]
    # baidu_secretKey = ['Z3kcAfpfFuwqvwN3Ixfd',
    #                    '_DQpFHMJVgYclCEkfNMZ']



    # 小牛翻译
    xiaoniu_apikeys = [#'42403ddf41072fd9771dddc13acc0900',  # bb
                       #'a40f0b3e5b40bdffa335b7d720cbd45f',  # bb
                       '5a979c2ec7a4756ede0cf63312f2a1b4',  # 卢聪
                       '288730da6848f9d4efa97c5fc64faf7a',  # cc
                       '67ae095d4ea80e2d3c8bc1ad479489e0',  # 老谭1
                       '940f33e891a6f6a0f2d9eb8c0da55b22',  # 老谭2
                       '99bbc6e59a9af7c4dda1c767feb62406',  # 杨伟龙
                       'f9f447aa36f8064092e717ad9d9b21b2',  # 梅阳
                       'cecab0774a12583effc646cf95376972',  # 瓴宇
                       'd4287120b398280f80ce638fd765480d',  # 真博
                       'eda25af81558904bd9d64034ecdc0924',  # 宇恒
                       'e1e1bc58768e1355ec2d65dcdbb268ca',  # 猴哥
                       '9493999482c68ef9a535d0a3d9eb02dc',  # 劲智
                       '960f0b40eca97a46ad594f24e7c4a555',  # 志飞
                       '517f50719f1b242e633538ff99f356ea',  # 贺云岳
                       '3de74e63611b05fb61308e58fbe43d6d',  # 张鹏乐
                       '75f1bcccca8590e936aca68c8c5055df',  # 脸盆
                       '4fbafe869644bc480f58dcecf8c08a9e',  # 瓴宇2
                       'd74813d54332b5278d3cf734fbdb0043'  # 瓴宇3
                       ]
    for i in range(len(xiaoniu_apikeys)):
        time0 = time.time()
        data_tmp = []
        for d in data:
            d_tmp = {
                'd': d,
                'is_xiaoniu': '1',
                'is_youdao': '0',
                'is_baidu': '0',
                'is_bai_pachong': '0',
                'apikey': xiaoniu_apikeys[i],
                'secretKey': '0'
            }
            data_tmp.append(d_tmp)
        data = data_tmp
        with Pool(processes=agents) as pool:
            result = pool.map(deal, data)
        data = result

        err_num = 0
        for d in data:
            if d['flag'] is False:
                err_num += 1

        with open(json_out, 'w') as file:
            json.dump(data, file)

        print('xiaoniu fanyi, time:%d, err_num:%d/%d' % (time.time()-time0, err_num, len(data)))

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

    json_in = '../data/train/train.json'
    json_out = '../data/train/train_c.json'
    trans(json_in, json_out)























