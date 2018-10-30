# coding = utf-8
# author = xy


import re
from data_pre.langconv import Converter


def deal_data(data):
    result = []
    for i in data:
        i = re.sub(r'\u3000', '', i)
        i = re.sub(r'\s+', ' ', i)

        # 全角数字 -> 半角数字
        i = re.sub(r'０', '0', i)
        i = re.sub(r'１', '1', i)
        i = re.sub(r'２', '2', i)
        i = re.sub(r'３', '3', i)
        i = re.sub(r'４', '4', i)
        i = re.sub(r'５', '5', i)
        i = re.sub(r'６', '6', i)
        i = re.sub(r'７', '7', i)
        i = re.sub(r'８', '8', i)
        i = re.sub(r'９', '9', i)

        # 全角字母 -> 半角字母 小写
        i = re.sub(r'ａ', 'a', i)
        i = re.sub(r'ｂ', 'b', i)
        i = re.sub(r'ｃ', 'c', i)
        i = re.sub(r'ｄ', 'd', i)
        i = re.sub(r'ｅ', 'e', i)
        i = re.sub(r'ｆ', 'f', i)
        i = re.sub(r'ｇ', 'g', i)
        i = re.sub(r'ｈ', 'h', i)
        i = re.sub(r'ｉ', 'i', i)
        i = re.sub(r'ｊ', 'j', i)
        i = re.sub(r'ｋ', 'k', i)
        i = re.sub(r'ｌ', 'l', i)
        i = re.sub(r'ｍ', 'm', i)
        i = re.sub(r'ｎ', 'n', i)
        i = re.sub(r'ｏ', 'o', i)
        i = re.sub(r'ｐ', 'p', i)
        i = re.sub(r'ｑ', 'q', i)
        i = re.sub(r'ｒ', 'r', i)
        i = re.sub(r'ｓ', 's', i)
        i = re.sub(r'ｔ', 't', i)
        i = re.sub(r'ｕ', 'u', i)
        i = re.sub(r'ｖ', 'v', i)
        i = re.sub(r'ｗ', 'w', i)
        i = re.sub(r'ｘ', 'x', i)
        i = re.sub(r'ｙ', 'y', i)
        i = re.sub(r'ｚ', 'z', i)

        # 全角字母 -> 半角字母 大写
        i = re.sub(r'Ａ', 'A', i)
        i = re.sub(r'Ｂ', 'B', i)
        i = re.sub(r'Ｃ', 'C', i)
        i = re.sub(r'Ｄ', 'D', i)
        i = re.sub(r'Ｅ', 'E', i)
        i = re.sub(r'Ｆ', 'F', i)
        i = re.sub(r'Ｇ', 'G', i)
        i = re.sub(r'Ｈ', 'H', i)
        i = re.sub(r'Ｉ', 'I', i)
        i = re.sub(r'Ｊ', 'J', i)
        i = re.sub(r'Ｋ', 'K', i)
        i = re.sub(r'Ｌ', 'L', i)
        i = re.sub(r'Ｍ', 'M', i)
        i = re.sub(r'Ｎ', 'N', i)
        i = re.sub(r'Ｏ', 'O', i)
        i = re.sub(r'Ｐ', 'P', i)
        i = re.sub(r'Ｑ', 'Q', i)
        i = re.sub(r'Ｒ', 'R', i)
        i = re.sub(r'Ｓ', 'S', i)
        i = re.sub(r'Ｔ', 'T', i)
        i = re.sub(r'Ｕ', 'U', i)
        i = re.sub(r'Ｖ', 'V', i)
        i = re.sub(r'Ｗ', 'W', i)
        i = re.sub(r'Ｘ', 'X', i)
        i = re.sub(r'Ｙ', 'Y', i)
        i = re.sub(r'Ｚ', 'Z', i)

        # 繁体 -> 简体
        i = Converter('zh-hans').convert(i)

        # 去除前后空格
        i = i.strip()

        result.append(i)

    return result
