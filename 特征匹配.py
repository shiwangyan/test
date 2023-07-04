# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('DSC_0001.JPG', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('example2.png', cv.IMREAD_GRAYSCALE)  # trainImage
#
# # Initiate ORB detector
# orb = cv.ORB_create()
#
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# # create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1, des2)
# # for match in matches:
# #     print(f"查询图像{match.queryIdx} \t和训练图像 {match.trainIdx} \t的距离是 {match.distance}")
# #
# # print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)
# # for match in matches:
# #     print(f"查询图像{match.queryIdx} \t和训练图像 {match.trainIdx} \t的距离是 {match.distance}")
#
#
# # Draw first 10 matches.
# img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# plt.imshow(img3), plt.show()


# ----------------------------------------------------------------


# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('DSC_0001.JPG', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('example2.png', cv.IMREAD_GRAYSCALE)  # trainImage
#
# # Initiate SIFT detector
# sift = cv.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
#
# # Apply ratio test
# good_match = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_match.append([m])
#
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# plt.imshow(img3), plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img1 = cv.imread('DSC_0001.JPG', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('example2.png', cv.IMREAD_GRAYSCALE)  # trainImage
#
# # Initiate SIFT detector
# sift = cv.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)  # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1, des2, k=2)
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in range(len(matches))]
#
# # ratio test as per Lowe's paper
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]
#
# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matchesMask,
#                    flags=cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# plt.imshow(img3, ), plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10
#
# img1 = cv.imread('DSC_0001.JPG', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('example2.png', cv.IMREAD_GRAYSCALE)  # trainImage
#
# # Initiate SIFT detector
# sift = cv.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
#
# flann = cv.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1, des2, k=2)
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good.append(m)
#
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h, w = img1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#
#     img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
# else:
#     print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
#     matchesMask = None
#
# draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                    singlePointColor=None,
#                    matchesMask=matchesMask,  # draw only inliers
#                    flags=2)
#
# img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#
# plt.imshow(img3, 'gray'), plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
#
# # Initiate ORB detector
# orb = cv.ORB_create()
#
# # find the keypoints with ORB
# kp = orb.detect(img, None)
#
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
#
# # draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
# plt.imshow(img2), plt.show()


# +++++++++++++++++++++++++++++++++++++
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
#
# # Initiate FAST detector
# star = cv.xfeatures2d.StarDetector_create()
#
# # Initiate BRIEF extractor
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
#
# # find the keypoints with STAR
# kp = star.detect(img, None)
#
# # compute the descriptors with BRIEF
# kp, des = brief.compute(img, kp)
#
# print(brief.descriptorSize())
# print(des.shape)


# +++++++++++++++++++++++++++++++++++

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)cd
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(500)

cv.destroyAllWindows()


