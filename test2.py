import requests


##
url = "http://10.255.0.19/drcom/login?callback=dr1003&DDDDD=2021201734%40unicom&upass=dwz142857&0MKKey=123456&R1=0&R3=0&R6=0&para=00&v6ip=&v=6000"
##
data = {
    'callback': 'dr1003',
    'DDDDD': '2021201734@unicom', #
    'upass': 'dwz142857', #
    '0MKKey': '123456',
    'R1': '0',
    'R3': '0',
    'R6': '0',
    'para': '00',
    'v6ip': '',
    'v': '6000'#
}

header = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',#
    'Connection': 'keep-alive',
    'Cookie': 'PHPSESSID=bn44jtctanqtba9m92l6ggv11d',#
    'Host': '10.255.0.19',
    'Referer': 'http://10.255.0.19/a79.htm',
    ##
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'
    ##
}

response = requests.post(url, data=data, headers=header).status_code
# 作用是向网页发送请求数据。需要提供网址，data和header三个数据
if response == 200:
    print("已经连上网啦，快去冲浪吧！")
else:
    print("连接失败，请检查一下配置吧！")
