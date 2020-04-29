# -*- coding: cp936 -*-

import socket
from time import ctime

bufsize = 1024
port = 8989
myname = socket.gethostname()
# 获取本机ip
myaddr = socket.gethostbyname(myname)
address = ("", port)

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind(address)
server_sock.listen(1)

print(myname, myaddr)
while True:
    print()
    'waiting for connection...'
    clientsock, addr = server_sock.accept()
    print('received from :', addr)

    while True:
        data = clientsock.recv(bufsize)
        print(' 收到---->%s\n%s' % (ctime(), data))
        clientsock.send(data)
    clientsock.close()

server_sock.close()
