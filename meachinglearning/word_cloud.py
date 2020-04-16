import socket

BUFSIZE = 4096
tcpServerSocket = socket.socket()  # 1.创建
hostip = '10.5.113.4'
port = 5678
tcpServerSocket.bind((hostip, port))  # 2.bind
tcpServerSocket.listen(5)  # 监听，设置等待队列最大数目
result = b''
i = 0
import base64
import hashlib
import json
import time

import requests


def comm(file_content):
    x_appid = '5ca32040'
    api_key = '76214aaa5687165154d0a9eec9538cab'
    curTime = str(int(time.time()))
    url = 'https://api.xfyun.cn/v1/service/v1/ise'
    text = "试卷内容"
    base64_audio = base64.b64encode(file_content)
    body = {'audio': base64_audio, 'text': text}
    param = json.dumps({"aue": "raw", "result_level": "entirety", "language": "zh_cn", "category": "read_chapter"})
    paramBase64 = str(base64.b64encode(param.encode('utf-8')), 'utf-8')
    m2 = hashlib.md5()
    m2.update((api_key + curTime + paramBase64).encode('utf-8'))
    checkSum = m2.hexdigest()
    x_header = {
        'X-Appid': x_appid,
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    req = requests.post(url, data=body, headers=x_header)
    result = req.content.decode('utf-8')
    print(result)
    return


while True:
    print("等待连接")
    clientSocket, addr = tcpServerSocket.accept()  # 3.接收连接请求，并获得ip和端口号
    while True:
        data = clientSocket.recv(BUFSIZE)  # 4.接收数据
        result += data
        # print(result)
        if not data:
            comm(result)
            print("break")
            break

        if not data:
            print("complete")

        s = "hello"
        clientSocket.send(s.encode())  # 5.发送数据
    clientSocket.close()
