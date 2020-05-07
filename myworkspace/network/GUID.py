#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QFont, QPicture
import myworkspace.network.QRCode as qcode


class VirtualMicWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.win_height = 250
        self.win_width = 400
        self.ip_name_qlablel = QLabel('IP:', self)
        self.ip_addr_alabel = QLabel('192.168.41.60', self)
        self.connect_status_alabel = QLabel('连接中...', self)
        self.qr_alabel = QLabel(self)
        self.initUI()

    def initUI(self):
        qfont = QFont()
        qfont.setPointSizeF(15.0)

        self.ip_name_qlablel.move(15, 50)
        self.ip_name_qlablel.setFont(qfont)

        self.ip_addr_alabel.move(15, 80)
        self.ip_addr_alabel.setFont(qfont)

        self.connect_status_alabel.move(15, 110)
        self.connect_status_alabel.setFont(qfont)

        self.setGeometry(400, 400, self.win_width, self.win_height)
        self.setWindowTitle('虚拟麦克风')
        self.show()

    def update_data(self):
        q = qcode.QRCodeGenerate()
        q.generate_qr_code()
        self.connect_status_alabel.setText("test")
        pixlmap = QPixmap("test.jpg")
        self.qr_alabel.move(self.win_width - pixlmap.width(), (self.win_height - pixlmap.height()) / 2)
        print(pixlmap.height())
        self.qr_alabel.setPixmap(pixlmap)
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VirtualMicWindow()
    ex.update_data()
    sys.exit(app.exec_())
