import numpy as np

matrix1 = np.asanyarray(
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
print(matrix1)
matrix2 = np.asanyarray([[1, 0, 1], [1, 0, 1], [1, 0, 1]])


# print(matrix2)


def get_patch(m, kernel_size, stride=1):
    m_len = len(m)

    length = m_len - kernel_size + 1
    if stride >= length:
        print('步长过长')
        ValueError('步长过长')
        return
    for row in range(0, length, stride):  # 按行遍历
        for column in range(0, length, stride):  # 按列遍历
            for j in range(row, length + row, 1):
                for k in range(column, length + column, 1):
                    print(m[j][k], end=' ')
                print()
            print('****************')
        print('----------------')


get_patch(matrix1, 3, 1)
