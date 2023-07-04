import cv2
import matplotlib.pyplot as plt


def find_card(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 对边缘进行膨胀操作
    dilated = cv2.dilate(edges, None, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    # 初始化牌的位置列表
    card_positions = []

    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 如果面积太小，则忽略
        if area < 1000:
            continue

        # 近似多边形轮廓
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果轮廓是四边形，则将其视为牌
        if len(approx) == 4:
            # 获取卡片的外接矩形
            x, y, w, h = cv2.boundingRect(approx)

            # 在原始图像上绘制矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 添加牌的位置到列表中
            card_positions.append((x, y, w, h))

    # 显示图像和中间结果
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].imshow(edges, cmap='gray', vmin=0, vmax=255)

    ax[1, 0].imshow(dilated, cmap='gray', vmin=0, vmax=255)


    plt.show()



    return card_positions


# 测试代码
image_path = 'example.png'
card_positions = find_card(image_path)
print("Card Positions:", card_positions)
