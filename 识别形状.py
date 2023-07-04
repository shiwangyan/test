import cv2
import numpy as np
import matplotlib.pyplot as plt


# 定义形状检测函数
def ShapeDetection(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
        perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
        if perimeter > 1000:
            cv2.drawContours(imgContour, obj, -1, (0, 255, 0), 2)  # 绘制轮廓线

        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
        CornerNum = len(approx)  # 轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度

        if area < 10000:
            continue

        if CornerNum == 4:
            objType = "Rectangle"
        else:
            continue

        cv2.rectangle(imgContour, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=5)  # 绘制边界框
        cv2.putText(imgContour, objType, (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0),
                    1)  # 绘制文字


# path = 'example.png'
path = 'example2.png'
img = cv2.imread(path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgContour = image_rgb.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
imgCanny = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
# 对边缘进行膨胀操作
dilated = cv2.dilate(imgCanny, None, iterations=5)
ShapeDetection(dilated)  # 形状检测

# 显示图像和中间结果
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0, 0].imshow(image_rgb, vmin=0, vmax=255)
ax[0, 1].imshow(imgGray, cmap='gray', vmin=0, vmax=255)
ax[0, 2].imshow(imgBlur, cmap='gray', vmin=0, vmax=255)

ax[1, 0].imshow(imgCanny, cmap='gray', vmin=0, vmax=255)
ax[1, 1].imshow(imgContour, cmap='gray', vmin=0, vmax=255)

plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=1)
#
# ax.imshow(imgContour, cmap='gray', vmin=0, vmax=255)
#
# plt.show()
