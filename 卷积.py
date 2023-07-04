import numpy as np


def convolve(image, kernel):
    # 获取图像和卷积核的形状
    height, width = image.shape[:2]
    k_height, k_width = kernel.shape[:2]

    # 创建一个空的输出图像
    output = np.zeros((height - k_height + 1, width - k_width + 1), dtype=np.float32)

    # 反转卷积核，以便进行卷积操作
    kernel = np.flipud(np.fliplr(kernel))

    # 迭代遍历输入图像并应用卷积核
    for y in range(k_height // 2, height - k_height // 2):
        for x in range(k_width // 2, width - k_width // 2):
            roi = image[y - k_height // 2:y + k_height // 2 + 1, x - k_width // 2:x + k_width // 2 + 1]
            k = (roi * kernel).sum()
            output[y - k_height // 2, x - k_width // 2] = k
    return output


if __name__ == '__main__':
    img = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])

    gaussian_kernel = (1.0 / 16) * np.array([[1, 2, 1],
                                             [2, 4, 2],
                                             [1, 2, 1]])

    convolve_img = convolve(img, gaussian_kernel)
    print(convolve_img)
