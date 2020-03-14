import socket

BUFSIZE = 4096
tcpServerSocket = socket.socket()  # 1.创建
hostip = '192.168.0.102'
port = 9999
tcpServerSocket.bind((hostip, port))  # 2.bind
tcpServerSocket.listen(5)  # 监听，设置等待队列最大数目
result = b''
i = 0

while True:
    print("等待连接")
    clientSocket, addr = tcpServerSocket.accept()  # 3.接收连接请求，并获得ip和端口号
    while True:
        data = clientSocket.recv(BUFSIZE)  # 4.接收数据
        result += data
        if not data:
            print("break")
            break

        if not data:
            with open("test.jpg", "wb") as f:
                f.write(result)
                print("complete")

        s = "hello"
        clientSocket.send(s.encode())  # 5.发送数据
    clientSocket.close()
