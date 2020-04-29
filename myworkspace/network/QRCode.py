import qrcode


class QRCodeGenerate():
    def __init__(self):
        print("construct")

    def generate_qr_code(self):
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=2)

        qr.add_data("url")
        qr.make(fit=True)
        img = qr.make_image()
        # img.show()
        # img.drawrect(10,10)
        img.save('test.jpg')


if __name__ == "__main__":
    q = QRCodeGenerate()
    q.generate_qr_code()
