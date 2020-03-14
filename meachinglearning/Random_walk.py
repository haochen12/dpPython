# 导入socket模块,时间模块
import socket  # socket模块是python自带的内置模块，不需要我们去下载
import time

# url为： http://images.gaga.me/photos2/2019/0416/5cb5e9950e25a.jpeg?watermark/1/image/aHR0cDovL3Jlcy5nYWdhLm1lL3dhdGVybWFyay9wYWl4aW4xLnBuZz9pbWFnZVZpZXcyLzIvdy80MDAvaC80MDA=/dissolve/50/gravity/Center/ws/1
# ip远程地址为：113.229.252.244
# 端口为：80
# 我们写的请求头如下：
http_req = b'''GET /photos2/2019/0416/5cb5e9950e25a.jpeg?watermark/1/image\
/aHR0cDovL3Jlcy5nYWdhLm1lL3dhdGVybWFyay9wYWl4aW4xLnBuZz9pbWFnZVZpZXcyLzIvdy80MDAvaC80MDA=\
/dissolve/50/gravity/Center/ws/1 HTTP/1.1\r\n\
Host: images.gaga.me\r\n\
User-Agent: Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0\r\n\r\n\
'''
# 建立对象
client = socket.socket()
# 连接,通过（ip,端口）来进行连接
client.connect(("192.168.0.102", 9999))
# 根据请求头来发送请求信息
client.send(http_req)
# 建立一个二进制对象用来存储我们得到的数据
result = b''
i = 0
# 得到响应数据
while True:
    # 每次获得的数据不超过1024字节
    http_resp = client.recv(1024)
    i = i + 1
    print("这是我们第{}次获得数据，获得的数据长度是{}，获得的数据内容为{}.".format(i, len(http_resp), http_resp))
    # 将每次获得的每行数据都添加到该对象中
    result += http_resp
    # 每获取一行数据便休眠一段时间，避免出现下次获得响应数据,因为速度太快，数据还未加载出来导致的我们获取不到数据的问题
    time.sleep(0.3)
    # 根据判断每一行获取的数据的字节长度来判断是否还存在数据，当该行数据长度等于0时，即已经获取到最后一行数据，终止循环
    if len(http_resp) <= 0:
        # 关闭浏览器对象
        client.close()
        # 终止循环
        break
# 由于我们获得的响应文件是包括响应头和图片信息两种的，而响应头是以\r\n\r\n来进行结尾的.
# 所以我们想获得图片信息可以以此来分割,又因为响应头是在前面的，所有我们只需要获得第二部分的图片即可
result = result.split(b"\r\n\r\n")[1]
print("我们获得的图片内容为{}.".format(result))
# 打开一个文件，将我们读取到的数据存入进去，即下载到本地我们获取到的图片
with open("可爱的小姐姐.jpg", "wb") as f:
    f.write(result)
