import numpy as np
import cv2


def cornerHarris(img, blocksize=2, ksize=3, k=0.04):
    # ksize是sobel的大小，blocksize是滑动窗口的大小
    def _clacHarris(cov, k):
        result = np.zeros([cov.shape[0], cov.shape[1]], dtype=np.float32)
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                a = cov[i, j, 0]
                b = cov[i, j, 1]
                c = cov[i, j, 2]
                result[i, j] = a * c - b * b - k * (a + c) * (a + c)
        return result

    Dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    Dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)

    cov = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cov[i, j, 0] = Dx[i, j] * Dx[i, j]
            cov[i, j, 1] = Dx[i, j] * Dx[i, j]
            cov[i, j, 2] = Dy[i, j] * Dy[i, j]
    # 算块内的梯度和，
    # 和[1,1
    #   1,1]相乘
    cov = cv2.boxFilter(cov, -1, (blocksize, blocksize), normalize=False)
    return _clacHarris(cov, k)


if __name__ == '__main__':
    image_name = "IMG_20230523_153337.jpg"
    image = cv2.imread(image_name)
    # 将图像转换为灰度图像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 知乎上的实现
    # result = cornerHarris(gray_img, blocksize=2, ksize=3, k=0.04)
    # pos = cv2.goodFeaturesToTrack(result, 0, 0.01, 10)
    # for i in range(len(pos)):
    #     cv2.circle(image, (int(pos[i][0][0]), int(pos[i][0][1])), 5, [255, 0, 0], thickness=3)




    # opencv 的实现
    # 计算Harris角点响应
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
    # 选择出最佳的角点
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    name_split = image_name.split(".")
    save_name = name_split[0] + "_harris." + name_split[1]

    cv2.imwrite(save_name, image)
    # cv2.imshow('harris', img)
    cv2.waitKey(0)



