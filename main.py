import time
import argparse
import sys
import numpy as np
import cv2 as cv
from icecream import ic
import copy

# Authors: Anton Goretsky, Daniel Mallia and Josh Winton
# Date begun: 5/14/2020
# Purpose:

# Referred to:
# https://docs.opencv.org/3.4/d5/dde/tutorial_feature_description.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

# Sample run
# python3 main.py --left_img=Datasets/Rui/dataset1/original_left/1.png --right_img=Datasets/Rui/dataset1/original_right/1.png --paper="2013"
# python3 main.py --left_img=Datasets/KITTI/Images/testing/image_2/000000_10.png --right_img=Datasets/KITTI/Images/testing/image_3/000000_10.png --paper="2013"


def read_images(lfp, rfp):
    img_l = cv.imread(lfp, 0)
    img_r = cv.imread(rfp, 0)

    if img_l is None:
        sys.exit("Could not open LEFT image file.")
    if img_r is None:
        sys.exit("Could not open RIGHT image file.")

    if(img_l.shape == img_r.shape):
        rows, cols = img_l.shape
        print("Rows: ", rows, " Cols: ", cols)
    else:
        sys.exit()

    return img_l, img_r


def view_keypoints(src, keypoints, orientation):
    img_with_keypoints = np.empty(src.shape, dtype=np.uint8)
    img_with_keypoints = cv.drawKeypoints(src, keypoints, img_with_keypoints)
    # cv.imshow("Image " + orientation, img_with_keypoints)
    cv.waitKey(0)


def check_keypoints(keypoints, orientation, minimum):
    num = len(keypoints)
    print("Number of keypoints calculated for ", orientation, ": ", num)
    if(num < minimum):
        print("Minimum number of keypoints not met for image: ", orientation)
        sys.exit()

# 2013 Algorithm
# Disparity calculation algorithm


def paper_2013(img_l, img_r, min_keypoints, min_matches):
    # Create detector and detect keypoints
    # May want to specify parameters for BRISK
    detector = cv.BRISK.create()
    keypoints_l, desc_l = detector.detectAndCompute(img_l, None)
    keypoints_r, desc_r = detector.detectAndCompute(img_r, None)

    # Print and check keypoint numbers
    check_keypoints(keypoints_l, "LEFT", min_keypoints)
    check_keypoints(keypoints_r, "RIGHT", min_keypoints)

    # Test view keypoints
    view_keypoints(img_l, keypoints_l, "LEFT")
    view_keypoints(img_r, keypoints_r, "RIGHT")

    # Borrowed from feature matching tutorial
    # Match keypoints - may want to change this to FLANN
    bfm = cv.BFMatcher(cv.NORM_HAMMING)
    # matches = bfm.match(desc_l, desc_r)
    matches = bfm.knnMatch(desc_l, desc_r, k=2)
    # matches = sorted(matches, key=lambda x: x.distance)
    print("Number of matches: ", len(matches))
    # img_matches = cv.drawMatches(img_l, keypoints_l, img_r, keypoints_r,
    #  matches[:1000], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imshow("Matches", img_matches)
    # cv.waitKey(0)

    # Get fundamental matrix
    good_matches = []
    for m, n in matches:
        if m.distance < 0.68*n.distance:
            good_matches.append(m)

    print(len(good_matches))
    img_matches = cv.drawMatches(img_l, keypoints_l, img_r, keypoints_r,
                                 good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Good Matches", img_matches)
    if len(good_matches) > min_matches:
        src_pts = np.float32(
            [keypoints_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        fund_mat, mask = cv.findFundamentalMat(src_pts, dst_pts)
        res, H1, H2 = cv.stereoRectifyUncalibrated(
            src_pts, dst_pts, fund_mat, img_l.shape, 10)
        img_l_transformed = cv.warpPerspective(
            src=img_l, M=H1, dsize=(img_l.shape[1], img_l.shape[0]))
        img_r_transformed = cv.warpPerspective(
            src=img_r, M=H2, dsize=(img_l.shape[1], img_l.shape[0]))
        # img_l_transformed=cv.warpPerspective(
        #     img_l, matrix, (img_l.shape[1], img_l.shape[0]))
        # src_pts = np.float32(
        #     [keypoints_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32(
        #     [keypoints_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # # May want to tune findFundamentalMat parameters
        # fund_mat, _ = cv.findFundamentalMat(src_pts, dst_pts)
        # H, _ = cv.findHomography(src_pts, dst_pts)
        # ic(H)
        # img_l_out = np.empty(img_l.shape, dtype=np.uint8)

        # fundamental = np.array([[float("6.06055943873370e-09"),	float("1.92223199610921e-06"), float("-0.000996332361126767")],
        #                        [float(
        #                            "-2.96185493980771e-06"), float("-9.80787000103649e-07"), float("0.174207730231353")],
        #                        [float("0.000903900940100796"), float("-0.172536484910197"), float("4.42844607271752")]], dtype=np.float32)

        # #cv.WARP_INVERSE_MAP = True
        # ic(img_l)
        # ic(fundamental)
        # #img_l_transformed = copy.deepcopy(img_l)
        # img_l_transformed = np.empty((img_l.shape[1], img_l.shape[0]), dtype=np.uint8)
        # img_l_transformed = cv.warpPerspective(src=img_l, M=H, dsize=(img_l.shape[1], img_l.shape[0]))
        # ic(img_l_transformed)
        # print(img_l.shape)
        # print(img_l_transformed.shape)
        cv.imshow("Transformed LEFT", img_l_transformed)
        cv.imshow("Transformed RIGHT", img_r_transformed)
        cv.imshow("left", img_l)
        cv.imshow("right", img_r)
        cv.waitKey(0)


def paper_2017():
    pass


def paper_2018():
    pass


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Disparity Map Estimation")
    parser.add_argument("--left_img", type=str, required=True,
                        help="Specify the filepath of the left image.")
    parser.add_argument("--right_img", type=str, required=True,
                        help="Specify the filepath of the right image.")
    parser.add_argument("--paper", type=str, required=True,
                        help="Specify the paper implementation.")
    parser.add_argument("--min_keypoints", type=int, required=False,
                        default=100, help="Specify the minimum number of keypoints")
    parser.add_argument("--min_matches", type=int, required=False, default=10,
                        help="Specify the minimum number of matches to proceed with 8-point\
			algorithm")
    args = parser.parse_args()

    # Read in images
    img_l, img_r = read_images(args.left_img, args.right_img)

    # 2013 paper
    paper_2013(img_l, img_r, args.min_keypoints, args.min_matches)


if __name__ == "__main__":
    main()
