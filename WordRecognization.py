import requests
import hmac
import hashlib
import base64
import time
import random
import re
import jieba
import pandas as pd
import argparse
import os
import string
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path',type=str, default="../data/scoring/imageUrl_ctr_add_category.csv",help="image ctr csv file")
parser.add_argument('--bucket_num',type=int,default=5,help="split images to {bucket_bum} class")
parser.add_argument('--text_info_dir',type=str,default="../data/text_info_new/",help="text info in images")
parser.add_argument('--images_dir',type=str,default="/data2/ericbkwang/images_categories/",help="dir of images by category")
args = parser.parse_args()

class WordRecognizer_on_Tencent_Cloud(object):
    """
    根据给出的 广告图片 image_url 调用腾讯云API得到其中的文字信息
    """
    def __init__(self):
        self.appid = "1257625747"
        self.bucket = "WordRecognization" #参考本文开头提供的链接
        self.secret_id = "AKIDQnFrRvfNKSUKyR4u3BCAxZ3LRcjlbFnB"  #参考官方文档
        self.secret_key = "McjjYCqX9e5NTN5bdu7GH7lC4pOn01rW"  #同上

        self.userid = "0"
        self.fileid = "tencentyunSignTest"
        self.url = "http://recognition.image.myqcloud.com/ocr/general"
        self._image_url = ""

    def set_image_url(self,image_url):
        self.expired = time.time() + 2592000
        self.onceExpired = 0
        self.current = time.time()
        self.rdm = ''.join(random.choice("0123456789") for i in range(10))
        self._init_info()
        self._init_sign()
        self._build_headers()
        self._image_url = image_url
        self._build_files()

    def _init_info(self):
        self.info = "a=" + self.appid + "&b=" + self.bucket + "&k=" + self.secret_id + "&e=" + str(self.expired) + "&t="\
                    + str(self.current) + "&r=" + str(self.rdm) + "&u=0&f="
    def _init_sign(self):
        self.signindex = hmac.new(bytes(self.secret_key,'utf-8'),bytes(self.info,'utf-8'), hashlib.sha1).digest()  # HMAC-SHA1加密
        self.sign = base64.b64encode(self.signindex + bytes(self.info,'utf-8'))  # base64转码，也可以用下面那行转码
        #sign=base64.b64encode(signindex+info.encode('utf-8'))
    def _build_headers(self):

        self.headers = {'Host': 'recognition.image.myqcloud.com',
                   "Authorization": self.sign,
                   }
    def _build_files(self):
        self.files = {'appid': (None,self.appid),
            'bucket': (None,self.bucket),
            # 'image': ('0.jpeg',open('/Users/eric/workspace/tencent/miaosi/0.jpeg','rb'),'image/jpeg'),
            'url': self._image_url
            }
    def request(self):
        self.r = requests.post(self.url, files=self.files,headers=self.headers)

        responseinfo = self.r.content
        data = responseinfo.decode('utf-8')

        r_index = r'itemstring":"(.*?)"'  # 做一个正则匹配
        result = re.findall(r_index, data)
        self._words = "".join(result)

    def get_content_TencentCloud(self):
        return self._words
class WordRecognizer_on_YouTu(object):
    def __init__(self):
        self.appid = "2108635500"
        self.appkey = "jYJjmbNukOMkPHbc"

    def curlmd5(self,src):
        m = hashlib.md5(src.encode('UTF-8'))
        # 将得到的MD5值所有字符转换成大写
        return m.hexdigest().upper()

    def get_params(self,base64_data):
        # 请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）
        t = time.time()
        time_stamp = str(int(t))
        # 请求随机字符串，用于保证签名不可预测
        nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
        # 应用标志，这里修改成自己的id和key
        params = {'app_id': self.appid,
                  'image': base64_data,
                  'time_stamp': time_stamp,
                  'nonce_str': nonce_str,
                  }
        sign_before = ''
        # 要对key排序再拼接
        for key in sorted(params):
            # 键值拼接过程value部分需要URL编码，URL编码算法用大写字母，例如%E8。quote默认大写。
            sign_before += '{}={}&'.format(key, quote(params[key], safe=''))
        # 将应用密钥以app_key为键名，拼接到字符串sign_before末尾
        sign_before += 'app_key={}'.format(self.appkey)
        # 对字符串sign_before进行MD5运算，得到接口请求签名
        sign = self.curlmd5(sign_before)
        params['sign'] = sign
        return params

    def get_content_YouTu(self,image_local_path):
        url = "https://api.ai.qq.com/fcgi-bin/ocr/ocr_generalocr"
        with open(image_local_path, 'rb') as fin:
            image_data = fin.read()
        base64_data = base64.b64encode(image_data)
        params = self.get_params(base64_data)
        r = requests.post(url, data=params)
        item_list = r.json()['data']['item_list']
        res = []
        for s in item_list:
            res.append(s['itemstring'])
        return "".join(res)
class ImageTextSaver(object):
    """
    将图片文字进行分类存放，分类的时候首先按照 一级行业 类目将所有的商品分类存放，然后在每个一级行业类目下分 bucket
    """
    def __init__(self,platform="Tencent_Cloud"):
        self.platform = platform
        self.bucket_num = args.bucket_num
        self.csv_path = args.csv_path
        self.images_dir = args.images_dir

        self.text_info_dir = args.text_info_dir  # get text info from image
        if not os.path.exists(self.text_info_dir): os.mkdir(self.text_info_dir)
        # id,url,crtSize,ctr,first_category,second_category
        self.data = pd.read_csv(self.csv_path)
        self.image_ids = self.data['id'].values.tolist()
        self.image_urls = self.data['url'].values.tolist()
        self.crtSizes = self.data['crtSize'].values.tolist()
        self.ctrs = self.data['ctr'].values.tolist()
        self.first_categories = self.data['first_category'].values.tolist()
        self.second_categories = self.data['second_category'].values.tolist()

        self.image_crtSize_ctrIntervals_map = {}  # 为每个crtSize都进行分桶，记录每个桶都间隔
        self._init_image_crtSize_ctrIntervals_map()

        # get text info from image url
        if platform=="Tencent_Cloud":
            self.wr = WordRecognizer_on_Tencent_Cloud()
        elif platform == "YouTu" or platform == "youtu":
            self.wr = WordRecognizer_on_YouTu()
    def _init_image_crtSize_ctrIntervals_map(self):
        """
        为每个crtSize都进行分桶，记录每个桶都间隔
        :return:
        """
        crtSize_CTRs_map = {}
        for i, i_crtSize in enumerate(self.crtSizes):
            if i_crtSize not in crtSize_CTRs_map:
                crtSize_CTRs_map[i_crtSize] = [self.ctrs[i]]
            else:
                crtSize_CTRs_map[i_crtSize].append(self.ctrs[i])

        # image_crtSize_ctrInterval_map: [crtSize1:[interval1, interval2, interval3, interval4, interval5],crtSize2:[],]
        for i_crtSize in crtSize_CTRs_map:
            self.image_crtSize_ctrIntervals_map[i_crtSize] = self._get_one_crtSize_ctrMap(crtSize_CTRs_map[i_crtSize])

    def _get_one_crtSize_ctrMap(self, this_ctrs):
        this_ctrs.sort()
        gap_length = len(this_ctrs) // self.bucket_num
        ctr_intervals = []

        for i in range(1, self.bucket_num):
            ctr_intervals.append(this_ctrs[gap_length * i])
        ctr_intervals.append(1)
        return ctr_intervals

    def ctr_bucket(self, i_crtSize, i_ctr):
        """
        根据 crtSize 和 CTR 得到bucket的index
        :param ctr:
        :return: 当前广告应该被分配到 哪个 bucket
        """
        ctr_intervals = self.image_crtSize_ctrIntervals_map[i_crtSize]
        for i, interval in enumerate(ctr_intervals):
            if i_ctr < interval:
                return i
            else:
                continue

    def run(self):
        try:
            for i, id in enumerate(self.image_ids):
                if id < 51149: continue
                # 获取当前 image 的crtSize 对应的bucket_no, 然后将图片复制到对应的文件夹中去
                i_image_url = self.image_urls[i]
                i_crtSize = self.crtSizes[i]
                i_ctr = self.ctrs[i]
                i_first_category = self.first_categories[i]
                i_image_local_path = os.path.join(self.images_dir,str(i_first_category),"{}.jpg".format(id))
                i_bucket_dir = os.path.join(self.text_info_dir,str(i_first_category), str(self.ctr_bucket(i_crtSize, i_ctr)))
                if not os.path.exists(i_bucket_dir): os.makedirs(i_bucket_dir)
                cmd = "wget {} -O {}.jpg".format(i_image_url, id)
                os.system(cmd)
                i_image_local_path = "./{}.jpg".format(id)
                if self.platform == "Tencent_Cloud":
                    self.wr.set_image_url(image_url=i_image_url)
                    self.wr.request()
                    sentence = self.wr.get_content_TencentCloud()
                elif self.platform == "YouTu" or self.platform == "youtu":
                    sentence = self.wr.get_content_YouTu(image_local_path=i_image_local_path)
                print(id, i_image_url)
                print(sentence)
                if sentence != "":
                    with open(os.path.join(i_bucket_dir,"{}.txt".format(id)), 'w') as file:
                        file.write(sentence)
                os.system("rm {}.jpg".format(id))
        except:
            print("image text save failed !")
        finally:
            print("Bucket Operation Done!")


if __name__ == '__main__':
    """
    wr = WordRecognizer()
    image_url = "http://pgdt.gtimg.cn/gdt/0/DAAYpeRAUAALQABiBbYRB0CvjbIdrz.jpg/0?ck=f8bbb29988a7b1151818703d7f74d79f"
    wr.set_image_url(image_url=image_url)
    wr.request()
    print(image_url)
    sentence = wr.get_words()
    print(sentence)
    exit()
    """
    image_text_saver = ImageTextSaver(platform="YouTu")
    image_text_saver.run()

