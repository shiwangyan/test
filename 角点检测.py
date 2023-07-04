import numpy as np

import matplotlib.pyplot as plt


def calculate_E(u, v, I, w):
    height, width = I.shape
    E = 0.0

    for x in range(width):
        for y in range(height):
            green_term = I[min(x + u, width - 1), min(y + v, height - 1)]
            red_term = I[x, y]
            diff = green_term - red_term
            squared_diff = diff ** 2
            weighted_squared_diff = w[x, y] * squared_diff
            E += weighted_squared_diff

    return E


if __name__ == '__main__':
    a = 0
    b = 255
    m = 50
    n = 50
    # 生成形状为(m, n)的随机整数矩阵，范围在[a, b]之间（包括a和b）
    matrix_img = np.random.randint(low=a, high=b + 1, size=(m, n))
    # matrix_window = np.random.randint(low=a, high=b + 1, size=(m, n))

    # calculate_E()

    # 生成形状为(m, n)的随机矩阵
    matrix = np.random.rand(m, n)

    # 显示矩阵
    # plt.imshow(matrix, cmap='viridis')
    plt.imshow(matrix, cmap='gray')
    plt.colorbar()
    plt.show()
