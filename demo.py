# import numpy as np
# import cv2
#
# # 创建一个大小为6x6的矩阵
# matrix = np.array([[70, 15, 30, 49, 5, 64],
#                    [7, 55, 3, 4, 16, 24],
#                    [1, 74, 48, 5, 39, 5],
#                    [8, 3, 65, 4, 17, 9],
#                    [45, 35, 8, 13, 4, 3],
#                    [7, 3, 25, 89, 4, 2]])
#
# # 创建一个大小为3x3的卷积核
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])
#
# # 对矩阵进行卷积操作，并使用cv2.convertScaleAbs()函数转换输出矩阵的深度和类型
# result = cv2.filter2D(matrix, -1, kernel)
# result = cv2.convertScaleAbs(result)
#
# # 显示卷积结果
# print('Convolution Result:\n', result)

# ----------------------------------
#
# import cv2
# import numpy as np
#
# # 原始矩阵
# matrix = np.array([[10, 20, 30, 40],
#                        [50, 60, 70, 80],
#                        [90, 100, 110, 120],
#                        [130, 140, 150, 160]
#                        ])
#
# # 定义填充宽度
# top, bottom, left, right = 1, 1, 1, 1
#
# # 使用copyMakeBorder函数进行填充
# BORDER_CONSTANT = cv2.copyMakeBorder(matrix, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)
# BORDER_REPLICATE = cv2.copyMakeBorder(matrix, top, bottom, left, right, cv2.BORDER_REPLICATE, value=1)
# BORDER_REFLECT = cv2.copyMakeBorder(matrix, top, bottom, left, right, cv2.BORDER_REFLECT, value=1)
# BORDER_WRAP = cv2.copyMakeBorder(matrix, top, bottom, left, right, cv2.BORDER_WRAP, value=1)
# BORDER_REFLECT_101 = cv2.copyMakeBorder(matrix, top, bottom, left, right, cv2.BORDER_REFLECT_101, value=1)
# """
# cv2.BORDER_CONSTANT: 添加一个恒定值的边框。
# cv2.BORDER_REPLICATE: 复制最近的像素值来填充边框。
# cv2.BORDER_REFLECT: 使用镜像反射来填充边框。
# cv2.BORDER_WRAP: 使用图像的另一侧来填充边框。
# cv2.BORDER_REFLECT_101：使用镜像反射（不包括边界像素）来填充边框。
# """
#
#
# # 打印结果
# print("Original Matrix:\n", matrix)
# print("BORDER_CONSTANT Matrix:\n", BORDER_CONSTANT)
# print("BORDER_REPLICATE Matrix:\n", BORDER_REPLICATE)
# print("BORDER_REFLECT Matrix:\n", BORDER_REFLECT)
# print("BORDER_WRAP Matrix:\n", BORDER_WRAP)
# print("BORDER_REFLECT_101 Matrix:\n", BORDER_REFLECT_101)


# -----------------------------------------------
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# G1 = np.array([[0.003, 0.013, 0.022, 0.013, 0.003],
#                [0.013, 0.059, 0.097, 0.059, 0.013],
#                [0.022, 0.097, 0.159, 0.097, 0.022],
#                [0.013, 0.059, 0.097, 0.059, 0.013],
#                [0.003, 0.013, 0.022, 0.013, 0.003]])
#
#
#
#
# G2 = np.array([[0.013, 0.025, 0.031, 0.025, 0.013],
#                [0.025, 0.059, 0.075, 0.059, 0.025],
#                [0.031, 0.075, 0.094, 0.075, 0.031],
#                [0.025, 0.059, 0.075, 0.059, 0.025],
#                [0.013, 0.025, 0.031, 0.025, 0.013]])
#
# # 定义高斯核尺寸和标准差
# ksize = (5, 5)
# sigma = 1
#
# # 生成高斯核矩阵
# G = cv2.getGaussianKernel(ksize=ksize[0], sigma=sigma, ktype=cv2.CV_64F)
# G2D = np.outer(G, G)
#
# # # 打印高斯核矩阵
# # print(G2D)
#
# # 创建一个 3 x 2 的子图布局，并在其中添加三张图像
# fig, ax = plt.subplots(nrows=2, ncols=2)
# # ax[0, 0].imshow(G1, cmap='gray', vmin=0, vmax=1)
# # ax[0, 1].imshow(G2, cmap='gray', vmin=0, vmax=1)
# ax[0, 0].imshow(G1, cmap='gray')
# ax[0, 1].imshow(G2, cmap='gray')
# ax[1, 0].imshow(G2D, cmap='gray')
#
# # ax[0, 1].imshow(edged_img, cmap='gray', vmin=0, vmax=255)
# # ax[1, 0].imshow(sharped_img, cmap='gray', vmin=0, vmax=255)
# plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
s2t.json Simplified Chinese to Traditional Chinese 簡體到繁體
t2s.json Traditional Chinese to Simplified Chinese 繁體到簡體
s2tw.json Simplified Chinese to Traditional Chinese (Taiwan Standard) 簡體到臺灣正體
tw2s.json Traditional Chinese (Taiwan Standard) to Simplified Chinese 臺灣正體到簡體
s2hk.json Simplified Chinese to Traditional Chinese (Hong Kong variant) 簡體到香港繁體
hk2s.json Traditional Chinese (Hong Kong variant) to Simplified Chinese 香港繁體到簡體
s2twp.json Simplified Chinese to Traditional Chinese (Taiwan Standard) with Taiwanese idiom 簡體到繁體（臺灣正體標準）並轉換爲臺灣常用詞彙
tw2sp.json Traditional Chinese (Taiwan Standard) to Simplified Chinese with Mainland Chinese idiom 繁體（臺灣正體標準）到簡體並轉換爲中國大陸常用詞彙
t2tw.json Traditional Chinese (OpenCC Standard) to Taiwan Standard 繁體（OpenCC 標準）到臺灣正體
hk2t.json Traditional Chinese (Hong Kong variant) to Traditional Chinese 香港繁體到繁體（OpenCC 標準）
t2hk.json Traditional Chinese (OpenCC Standard) to Hong Kong variant 繁體（OpenCC 標準）到香港繁體
t2jp.json Traditional Chinese Characters (Kyūjitai) to New Japanese Kanji (Shinjitai) 繁體（OpenCC 標準，舊字體）到日文新字體
jp2t.json New Japanese Kanji (Shinjitai) to Traditional Chinese Characters (Kyūjitai) 日文新字體到繁體（OpenCC 標準，舊字體）
tw2t.json Traditional Chinese (Taiwan standard) to Traditional Chinese 臺灣正體到繁體（OpenCC 標準）
"""

# -*- coding: utf8 -*-
import opencc
import os

# 创建OpenCC转换器
converter = opencc.OpenCC('t2s.json')

input_name = "基于图像三维重建.2.课前介绍.srt"
file_name, file_format = os.path.splitext(input_name)
# file_name, file_format = input_name.split(".")
output_name = file_name + "_translated" + "." + file_format
# 读取输入文件内容
with open(input_name, 'r', encoding='utf-8') as f:
    content = f.read()

# 转换为简体中文
converted_content = converter.convert(content)

# 将转换后的内容保存到输出文件中
with open(output_name, 'w', encoding='utf-8') as f:
    f.write(converted_content)
