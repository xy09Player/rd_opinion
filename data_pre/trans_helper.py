import http.client
import hashlib
import json
import urllib, urllib.request
import random
from urllib.parse import quote
from urllib.request import urlopen
from urllib import request, parse
import requests


def baidu_translate(content, fromLang, toLang, appid, secretKey):
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = content
    # fromLang = 'zh' # 源语言
    # toLang = 'en'   # 翻译后的语言
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")# 获得返回的结果，结果为json格式
        js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
        dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
        return dst.strip(), 0
    except Exception as e:
        return 'xxx', 1
    finally:
        if httpClient:
            httpClient.close()


def xiaoniu_translate(content, fromLang, toLang, apikey='42403ddf41072fd9771dddc13acc0900'):
    # apikey 是bb的

    host = 'http://api.niutrans.vip'
    path = '/NiuTransServer/translation'
    querys = 'from=' + fromLang + '&to=' + toLang + '&src_text=' + urllib.parse.quote(content) + '&apikey=' + apikey
    url = host + path + '?' + querys
    try:
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        res = response.read().decode('utf-8')
        js = json.loads(res)  # 将json格式的结果转换字典结构
        dst = str(js["tgt_text"])  # 取得翻译后的文本结果
        return dst.strip(), 0
    except Exception as e:
        # print(len(content))
        return 'xxx', 1


def youdao_translate(content, fromLang, toLang, apikey='186483e9dd22f401', secretKey='Y1mX21LrPIVR7VT41wBQLz6BqyPfd4Mu'):
    httpClient = None
    myurl = '/api'
    q = content
    # fromLang = 'EN'
    # toLang = 'zh-CHS'
    salt = random.randint(1, 65536)

    sign = apikey+q+str(salt)+secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()

    myurl = myurl+'?appKey='+apikey+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

    try:
        httpClient = http.client.HTTPConnection('openapi.youdao.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")# 获得返回的结果，结果为json格式
        js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
        dst = str(js["translation"][0])  # 取得翻译后的文本结果
        return dst.strip(), 0
    except Exception as e:
        print(e)
        return 'xxx', 1
    finally:
        if httpClient:
            httpClient.close()


class BaiFanyi():

    def __init__(self,trant_str):
        self.trant_str = trant_str
        self.lan_url = "https://fanyi.baidu.com/langdetect"
        self.trant_url = "https://fanyi.baidu.com/basetrans"
        self.headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3528.4 Mobile Safari/537.36"}

    def __parse_url(self,url,data):
        reponse = requests.post(url,data=data,headers =self.headers)
        return json.loads(reponse.content.decode())

    def __trant_url(self,url,data):
        reponse_trant_url = requests.post(url,data=data,headers=self.headers)
        return json.loads(reponse_trant_url.content.decode())

    def __show_reslut(self,result):
        show = result['trans'][0]['dst']
        return show

    def run(self):
        # 1、获取翻译类型
            # 1.1准备post的url地址，post_data
            post_data = {"query":self.trant_str}
            # 1.2发送post请求，获取相应
            lan_result = self.__parse_url(self.lan_url,post_data)
            # 1.3提取语言类型
            lan = lan_result['lan']

        #2、准备post数据
            trant_data = {"query": self.trant_str, "from": "zh", "to": "en"} if lan == "zh" else {"query": self.trant_str, "from": "en", "to": "zh"}
        # 3发送请求、获取相应
            trant_result = self.__trant_url(self.trant_url,trant_data)

        # 4、提取翻译的结果
            return self.__show_reslut(trant_result)





if __name__ == '__main__':
    content = '国际及地区航班1、乘坐从中国境内机场始发的国际'
    # content = 'International and regional flights 1, departing from airports in China'
    print(xiaoniu_translate(content, 'zh', 'en', 'a78bf9cbdbd4d2d1c5b152382d963a7c'))


















