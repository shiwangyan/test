import cv2
import numpy as np

# 读取图像
image_name = "IMG_20230508_163007.jpg"
image = cv2.imread(image_name)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点和描述符
keypoints, descriptors = sift.detectAndCompute(image, None)
# Here kp will be a list of keypoints and des is a numpy array of shape (Number of Keypoints)×128.
# 这里的 keypoints 是一个关键点列表，descriptors 是一个形状为 （关键点数 × 128） 的 numpy 数组。

# kp = keypoints[0]
# print(dir(kp))
# print("位置：", kp.pt)
# print("尺度：", kp.size)
# print("方向：", kp.angle)

# for kp in keypoints:
#     print("位置：", kp.pt)
#     print("尺度：", kp.size)
#     print("方向：", kp.angle)


# print(f"keypoints:{keypoints}")
# print(f"descriptors:{descriptors}")


# 绘制关键点
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 绘制特征点框
# for kp in keypoints:
#     x, y = kp.pt
#     size = kp.size
#     angle = kp.angle
#     rect_points = cv2.boxPoints(((x, y), (size, size), angle))
#     rect_points = np.int0(rect_points)
#     cv2.drawContours(image, [rect_points], 0, (0, 255, 0), 2)

# 显示图像
name_split = image_name.split(".")
save_name = name_split[0] + "_sift." + name_split[1]

# cv2.imwrite(save_name, image)
cv2.imwrite(save_name, image_with_keypoints)
# cv2.imshow("Image with Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
