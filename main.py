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
# python3 main.py --left_img=Datasets/KITTI2012/testing/image_0/000003_10.png --right_img=Datasets/KITTI2012/testing/image_1/000003_10.png --paper="2013"


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


def left_right_disparity_2013(padded_img_l, padded_img_r, shape, pad, dmax, tau):
    min_disparity = shape[1]
    max_disparity = 0
    disp = [[0] * shape[1] for i in range(shape[0])]
    v_max = shape[0] + pad - 1
    u_max = shape[1] + pad - 1
    # V_max disparity calculation
    for u in range(pad, u_max + 1):
        # Create "template"
        template = padded_img_l[v_max-pad:v_max+pad+1, u-pad:u+pad+1]
        # Select "image"
        image_for_search = padded_img_r[v_max -
                                        pad:v_max+pad+1, max(u-dmax-pad, pad):u+pad+1]
        # calculate disparity for each pixel
        result = cv.matchTemplate(
            image_for_search, template, method=cv.TM_CCORR_NORMED)
        _, _, _, index = cv.minMaxLoc(result)
        # minMaxLoc returns the x (col), y (row) of the max number in the array
        # here we just want the column, but we need to subtract from the length
        # to convert from index to disparity
        curr_disp = (result.shape[1]-1) - index[0]
        if(curr_disp > max_disparity):
            max_disparity = curr_disp
        elif(curr_disp < min_disparity):
            min_disparity = curr_disp
        disp[v_max-pad][u-pad] = curr_disp

    # for remaining rows
    for v in range(v_max-1, pad-1, -1):
        # for each column
        for u in range(pad, u_max + 1):
            search_range = {}
            # all ranges must have 1 added to the right term for inclusion
            # get down left search range
            if(u-pad-1 >= 0):
                retrieved_d = disp[v-pad+1][u-pad-1]
                for i in range(
                        retrieved_d-tau, retrieved_d+tau+1):
                    if i not in search_range and i >= 0:
                        search_range[i] = i
            # get down search range
            retrieved_d = disp[v-pad+1][u-pad]
            for i in range(
                    retrieved_d-tau, retrieved_d+tau+1):
                if i not in search_range and i >= 0:
                    search_range[i] = i
            # get down right search range
            if(u-pad+1 < shape[1]):
                retrieved_d = disp[v-pad+1][u-pad+1]
                for i in range(
                        retrieved_d-tau, retrieved_d+tau+1):
                    if i not in search_range and i >= 0:
                        search_range[i] = i
            # Create "template"
            template = padded_img_l[v-pad:v+pad+1, u-pad:u+pad+1]
            # Iterate through image segments and calculate NCC
            highest_correlation = -1
            highest_correlation_disparity = 0
            for d in search_range:
                if(u-d-pad >= 0):
                    image_for_search = padded_img_r[v-pad:v+pad+1,
                                                    u-d-pad:u-d+pad+1]
                    result = cv.matchTemplate(image_for_search, template,
                                              method=cv.TM_CCORR_NORMED)[0][0]
                    if(result > highest_correlation):
                        highest_correlation = result
                        highest_correlation_disparity = d

            disp[v-pad][u-pad] = highest_correlation_disparity
            if(highest_correlation_disparity > max_disparity):
                max_disparity = highest_correlation_disparity
            elif(highest_correlation_disparity < min_disparity):
                min_disparity = highest_correlation_disparity

    return disp, min_disparity, max_disparity


def right_left_disparity_2013(padded_img_l, padded_img_r, shape, pad, dmax, tau):
    min_disparity = shape[1]
    max_disparity = 0
    disp = [[0] * shape[1] for i in range(shape[0])]
    v_max = shape[0] + pad - 1
    u_max = shape[1] + pad - 1
    # V_max disparity calculation
    for u in range(pad, u_max + 1):
        # Create "template"
        # CHANGED  (changed img_l to img_r)
        template = padded_img_r[v_max-pad:v_max+pad+1, u-pad:u+pad+1]
        # Select "image"
        image_for_search = padded_img_l[v_max-pad:v_max+pad+1,  # CHANGED (changed img_r to img_l, changed from [v_max - pad:v_max+pad+1, max(u-dmax-pad, pad):u+pad+1]
                                        u-pad:min(u+dmax+pad+1, u_max+pad+1)]
        # calculate disparity for each pixel
        result = cv.matchTemplate(
            image_for_search, template, method=cv.TM_CCORR_NORMED)
        _, _, _, index = cv.minMaxLoc(result)
        # minMaxLoc returns the x (col), y (row) of the max number in the array
        # here we just want the column, but we need to subtract from the length
        # to convert from index to disparity
        # CHANGED (changed from (result.shape[1] - 1) - index[0])
        curr_disp = index[0]
        if(curr_disp > max_disparity):
            max_disparity = curr_disp
        elif(curr_disp < min_disparity):
            min_disparity = curr_disp
        disp[v_max-pad][u-pad] = curr_disp

    # for remaining rows
    for v in range(v_max-1, pad-1, -1):
        # for each column
        for u in range(pad, u_max + 1):
            search_range = {}
            # all ranges must have 1 added to the right term for inclusion
            # get down left search range
            if(u-pad-1 >= 0):
                retrieved_d = disp[v-pad+1][u-pad-1]
                for i in range(
                        retrieved_d-tau, retrieved_d+tau+1):
                    if i not in search_range and i >= 0:
                        search_range[i] = i
            # get down search range
            retrieved_d = disp[v-pad+1][u-pad]
            for i in range(
                    retrieved_d-tau, retrieved_d+tau+1):
                if i not in search_range and i >= 0:
                    search_range[i] = i
            # get down right search range
            if(u-pad+1 < shape[1]):
                retrieved_d = disp[v-pad+1][u-pad+1]
                for i in range(
                        retrieved_d-tau, retrieved_d+tau+1):
                    if i not in search_range and i >= 0:
                        search_range[i] = i
            # Create "template"
            # CHANGED (changed from img_l to img_r)
            template = padded_img_r[v-pad:v+pad+1, u-pad:u+pad+1]
            # Iterate through image segments and calculate NCC
            highest_correlation = -1
            highest_correlation_disparity = 0
            for d in search_range:
                if(u+d+pad <= u_max+pad):  # CHANGED     (changed from u-d-pad>= 0)
                    image_for_search = padded_img_l[v-pad:v+pad+1,  # CHANGED  (changed from image_for_search = padded_img_r[v-pad:v+pad+1,u-d-pad:u-d+pad+1])
                                                    u+d-pad:u+d+pad+1]  # CHANGED (marked above)
                    result = cv.matchTemplate(image_for_search, template,
                                              method=cv.TM_CCORR_NORMED)[0][0]
                    if(result > highest_correlation):
                        highest_correlation = result
                        highest_correlation_disparity = d

            disp[v-pad][u-pad] = highest_correlation_disparity
            if(highest_correlation_disparity > max_disparity):
                max_disparity = highest_correlation_disparity
            elif(highest_correlation_disparity < min_disparity):
                min_disparity = highest_correlation_disparity

    return disp, min_disparity, max_disparity


def normalize_disp_map(disp_map, min_disparity, max_disparity):
    for i in range(len(disp_map)):
        for j in range(len(disp_map[0])):
            disp_map[i][j] = ((disp_map[i][j] - min_disparity) /
                              (max_disparity - min_disparity)) * 255


def left_right_consistency_check_2017(disp_l, disp_r, threshold):
    max_disparity = 0
    disp_l_checked = copy.deepcopy(disp_l)
    for i in range(len(disp_l)):
        for j in range(len(disp_l[0])):
            d_l = disp_l[i][j]
            offset = j - d_l
            # if (offset >= 0):
            d_r = disp_r[i][offset]
            if abs(d_l - d_r) > threshold:
                disp_l_checked[i][j] = 0

            if(d_l > max_disparity):
                max_disparity = d_l

    return disp_l_checked, max_disparity

# 2013 Algorithm
# Disparity calculation algorithm
# Assumes rectified input images


def paper_2013(img_l, img_r, w, dmax, tau, disp_thresh):
    start = time.time()
    # Algorithm 1
    # NOTE: The paper mixes up u and v - u is specified as row and v is
    # specified as column number, but they then use the reverse. We have
    # kept the labels but reversed the order for proper (row, col) order.
    # "Vmax" - bottom row

    # The algorithm does not specify how to properly handle the template window
    # size with the maximum row. Here we have zero-padded for proper handling.
    pad = int((w-1) / 2)
    shape = img_l.shape
    padded_shape = (shape[0] + 2*pad, shape[1] + 2*pad)
    padded_img_l = np.zeros(padded_shape, dtype=np.uint8)
    padded_img_l[pad:pad + shape[0], pad:pad + shape[1]] = img_l
    padded_img_r = np.zeros(padded_shape, dtype=np.uint8)
    padded_img_r[pad:pad + shape[0], pad:pad + shape[1]] = img_r

    # Calculate disparity maps
    disp_l, l_min, l_max = left_right_disparity_2013(padded_img_l, padded_img_r, shape, pad,
                                                     dmax, tau)
    disp_r, r_min, r_max = right_left_disparity_2013(padded_img_l, padded_img_r, shape, pad,
                                                     dmax, tau)

    # Left-right consistency check
    disp_l_checked, max_disparity_checked = left_right_consistency_check_2017(
        disp_l, disp_r, disp_thresh)

    # Normalize disparity maps
    normalize_disp_map(disp_l, l_min, l_max)
    normalize_disp_map(disp_r, r_min, r_max)
    normalize_disp_map(disp_l_checked, 0, max_disparity_checked)

    # Write disparity maps
    cv.imwrite("left_disp_map.png", np.array(disp_l, dtype=np.uint8))
    cv.imwrite("right_disp_map.png", np.array(disp_r, dtype=np.uint8))

    # Write checked disparity map
    cv.imwrite("left_right_checked.png", np.array(
        disp_l_checked, dtype=np.uint8))

    end = time.time()
    print("Time elapsed (seconds): ", end - start)

def mean_map_calculation(img, int_img, w):
    mean_map_left = [[0] * img.shape[1] for i in range(img.shape[0])]
    rho = int((w-1)/2)
    m = img.shape[0]
    n = img.shape[1]

    max_mean = -np.inf
    min_mean = np.inf

    for v in range(rho, m - rho):
        for u in range(rho, n - rho):
            r1 = int_img[v+rho+1, u+rho+1]
            r2 = int_img[v-rho, u-rho]
            r3 = int_img[v+rho+1, u-rho]
            r4 = int_img[v-rho, u+rho+1]
            value = (r1 + r2 - r3 - r4) / (w ** 2)
            mean_map_left[v][u] = value
            max_mean = max(max_mean, value)
            min_mean = min(min_mean, value)

    return mean_map_left, min_mean, max_mean

def sigma_map_calculation(int_img2, mean_map, w):
    

def paper_2017(img_l, img_r, w, d_max, tau, disp_thresh):
    start = time.time()
    # alg 1: int image init
    # data: left and right integral images with dimensions m+1 by n+1
    # result: integral image of size m+1 x n+1
    # NOTE - see: https://stackoverflow.com/questions/30195420/why-the-integral-image-contains-extra-row-and-column-of-zeros
    int_img_l, int_img_l2 = cv.integral2(img_l)
    int_img_r, int_img_r2 = cv.integral2(img_r)

    # alg 2: left disparity map estimation
    # data: left + right image, mean, sigma
    # result: left disp map

    ################ MEAN MAP CALCULATION ####################
    mean_map_left, l_min_mean , l_max_mean = mean_map_calculation(img_l, int_img_l, w)
    mean_map_right, r_min_mean , r_max_mean = mean_map_calculation(img_r, int_img_r, w)

    #normalize_disp_map(mean_map_left, l_min_mean, l_max_mean)
    #normalize_disp_map(mean_map_right, r_min_mean, r_max_mean)

    #cv.imwrite("left_mu_map.png", np.array(mean_map_left, dtype=np.float32))
    #cv.imwrite("right_mu_map.png", np.array(mean_map_right, dtype=np.float32))

    ########## SIGMA CALCULATION ###########
    sigma_map_left = [[0] * int_img_l.shape[1] for i in range(int_img_l.shape[0])]

    max_sigma = -np.inf
    min_sigma = np.inf

    for v in range(rho, m-rho):
        for u in range(rho, n-rho):
            sum = 0
            for W_v in range(v-rho, v+rho+1):
                for W_u in range(u-rho, u+rho+1):
                    i_l = img_l[W_v][W_u]
                    mean_l = mean_map_left[W_v][W_u]
                    sum += float((i_l - mean_l) ** 2)/(w ** 2)
            value = np.sqrt(sum)
            sigma_map_left[v][u] = value
            max_sigma = max(max_sigma, value)
            min_sigma = min(min_sigma, value)

    for i in range(len(sigma_map_left)):
        for j in range(len(sigma_map_left[0])):
            sigma_map_left[i][j] = (
                (sigma_map_left[i][j] - min_sigma) / (max_sigma - min_sigma)) * 255

    cv.imwrite("left_sigma_map.png", np.array(sigma_map_left, dtype=np.float32))

    ########## RIGHT Sigma BLOCK Calculation ###########
    # sigma_map_right = np.zeros(int_img_r.shape, dtype=np.float32)

    # max_sigma = -np.inf
    # min_sigma = np.inf

    # for v in range(rho, m-rho):
    #     for u in range(rho, n-rho):
    #         for W_v in range(v-rho, v+rho):
    #             for W_u in range(u-rho, u+rho):
    #                 i_r = img_r[W_v][W_u]
    #                 mean_r = mean_map_right[W_v][W_u]
    #                 sigma_map_right[v][u] = np.sqrt(((i_r - mean_r) ** 2)/(w ** 2))
    #                 max_sigma = max(max_sigma, sigma_map_right[v][u])
    #                 min_sigma = min(min_sigma, sigma_map_right[v][u])

    # for i in range(len(sigma_map_right)):
    #     for j in range(len(sigma_map_right[0])):
    #         sigma_map_right[i][j] = ((sigma_map_right[i][j] - min_sigma) / (max_sigma - min_sigma)) * 255

    # cv.imwrite("sigma_map.png", sigma_map_right)

    # v = m - rho - 2
    # for u_l in range(rho + d_max+1, n-rho-d_max-1):
    #     for d in range(0, 71):
    #         u_r = u_l - d
    #         if

    end = time.time()
    print("Time elapsed (seconds): ", end - start)


def paper_2018():
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
    matches = bfm.match(desc_l, desc_r)
    # matches = bfm.knnMatch(desc_l, desc_r, k=2)
    # ic(matches)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Number of matches: ", len(matches))
    img_matches = cv.drawMatches(img_l, keypoints_l, img_r, keypoints_r,
                                 matches[:1000], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matches", img_matches)
    # cv.waitKey(0)

    # Get fundamental matrix
    good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good_matches.append(m)
    for m in matches:
        # print(m.distance)
        if m.distance < 1000:
            good_matches.append(m)

    if len(good_matches) > min_matches:
        src_pts = []
        dst_pts = []
        for i in range(4):
            src_pts.append(list(keypoints_l[good_matches[i].queryIdx].pt))
            dst_pts.append(list(keypoints_r[good_matches[i].trainIdx].pt))

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)

        ic(type(src_pts))
        ic(src_pts)

        # topLeftDest = [0, 0]
        # dest = np.float32([topLeftDest,bottomLeftDest, topRightDest, \
        #         bottomRightDest])

        matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
        img_l_transformed = cv.warpPerspective(
            img_l, matrix, (img_l.shape[1], img_l.shape[0]))
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
        cv.imshow("left", img_l)
        cv.imshow("right", img_r)
        cv.waitKey(0)


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
    parser.add_argument("--w", type=int, required=False, default=5,
                        help="Specify the window size.")
    parser.add_argument("--dmax", type=int, required=False, default=100,
                        help="Specify the maximum disparity search range.")
    parser.add_argument("--tau", type=int, required=False, default=2,
                        help="Specify the tau (tolerance).")
    parser.add_argument("--disp_thresh", type=int, required=False, default=5,
                        help="Specify the left-right consistency check absolute difference \
        threshold")
    args = parser.parse_args()

    # Read in images
    img_l, img_r = read_images(args.left_img, args.right_img)

    # 2013 paper
    if(args.paper == "2013"):
        paper_2013(img_l, img_r, args.w, args.dmax, args.tau, args.disp_thresh)
    elif(args.paper == "2017"):
        paper_2017(img_l, img_r, args.w, args.dmax, args.tau, args.disp_thresh)


if __name__ == "__main__":
    main()
