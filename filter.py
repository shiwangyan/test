import cv2
import numpy as np
import matplotlib.pyplot as plt


def average_kernel(size):
    """
    创建一个大小为（size，size）的平均滤波核矩阵
    :param size: 核的大小
    :return: 平均滤波核矩阵
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= size * size

    return kernel


if __name__ == '__main__':
    # 读取图像
    img = cv2.imread("sample.png")
    # matrix = np.array([[10, 15, 30, 49, 5, 64],
    #                    [7, 55, 3, 4, 16, 24],
    #                    [1, 74, 148, 5, 39, 5],
    #                    [8, 3, 165, 4, 117, 9],
    #                    [45, 35, 8, 13, 24, 3],
    #                    [71, 3, 25, 89, 4, 2]
    #                    ], dtype="uint8")
    matrix = np.array([[10, 20, 30, 40],
                       [50, 60, 70, 80],
                       [90, 100, 110, 120],
                       [130, 140, 150, 160]
                       ], dtype="float32")

    shift = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 0, 0]])

    # 创建3x3大小的平均滤波器
    kernel_size = 3
    kernel_3 = average_kernel(kernel_size)
    kernel_5 = average_kernel(5)
    kernel_7 = average_kernel(7)

    # 对图像进行滤波
    # filtered_img = cv2.filter2D(matrix, ddepth=-1, kernel=shift, borderType=cv2.BORDER_REFLECT_101)
    smoothed_img_3 = cv2.filter2D(img, ddepth=-1, kernel=kernel_3, borderType=cv2.BORDER_REFLECT_101)
    smoothed_img_5 = cv2.filter2D(img, ddepth=-1, kernel=kernel_5, borderType=cv2.BORDER_REFLECT_101)
    smoothed_img_7 = cv2.filter2D(img, ddepth=-1, kernel=kernel_7, borderType=cv2.BORDER_REFLECT_101)
    """
    cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) → dst
    src: 输入图像，可以是单通道或多通道图像。
    ddepth: (desired depth)输出图像深度，通常使用-1表示与输入图像相同的深度。如果需要更高精度，则可以使用cv2.CV_32F。
    kernel: 卷积核，可以是任何形状和尺寸的数组，但必须是单通道浮点类型。
    dst（可选）：输出图像，大小和深度与输入图像相同。
    anchor（可选）：核函数锚点，默认为(-1, -1)，表示核函数中心点。
    delta（可选）：可选的偏移量，用于调整输出图像的亮度值。
    borderType（可选）：边界处理方法，可以是以下几种之一：
        cv2.BORDER_CONSTANT: 添加一个恒定值的边框。
        cv2.BORDER_REPLICATE: 复制最近的像素值来填充边框。
        cv2.BORDER_REFLECT: 使用镜像反射来填充边框。
        cv2.BORDER_WRAP: 使用图像的另一侧来填充边框。
        cv2.BORDER_REFLECT_101：使用镜像反射（不包括边界像素）来填充边框。
    """
    # print(matrix)
    # print(filtered_img)

    # edged_img = img - smoothed_img
    #
    # sharped_img = img + edged_img

    # 显示原始图像和滤波后的图像
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Smoothed Image', smoothed_img)
    # cv2.imshow('Edged Image', edged_img)
    # cv2.imshow('Sharped Image', sharped_img)
    # cv2.waitKey(0)



    # 创建一个 3 x 2 的子图布局，并在其中添加三张图像
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].imshow(smoothed_img_3, cmap='gray', vmin=0, vmax=255)
    ax[1, 0].imshow(smoothed_img_5, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].imshow(smoothed_img_7, cmap='gray', vmin=0, vmax=255)
    # ax[0, 1].imshow(edged_img, cmap='gray', vmin=0, vmax=255)
    # ax[1, 0].imshow(sharped_img, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # plt.imshow(matrix, cmap='gray', vmin=0, vmax=255)
    # plt.colorbar()
    # plt.show()
