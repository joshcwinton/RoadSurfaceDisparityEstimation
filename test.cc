#include <stdio.h>
#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void paper_2013(Mat img_l, Mat img_r, const int w, const int dmax, const int tau) {
	const int pad = int((w-1)/2);
	int rows = img_l.rows;
	int cols = img_l.cols;
	int padded_rows = rows + 2*pad;
	int padded_cols = cols + 2*pad;
	Mat padded_img_l(padded_rows, padded_cols, CV_8UC1);
	Mat padded_img_r(padded_rows, padded_cols, CV_8UC1);
	copyMakeBorder(img_l, padded_img_l, pad, pad, pad, pad, BORDER_CONSTANT, 0);
	copyMakeBorder(img_r, padded_img_r, pad, pad, pad, pad, BORDER_CONSTANT, 0);

	Mat disparities(rows, cols, CV_8UC1);
	int v_max = rows + pad - 1;
	int u_max = cols + pad - 1;

	// Bottom row
	int top = v_max-pad;
	for(int u = pad; u < u_max + 1; u++) {
		// Template
		Rect temp_param(u-pad, top, w, w);
		Mat search_template(padded_img_r, temp_param);

		// Image for search
		Point tl, br;
		tl.x = std::max(u-dmax-pad, 0);
		tl.y = top;
		br.x = u + pad;
		br.y = top + w;
		Rect img_param(tl, br);
		Mat image_for_search(padded_img_r, img_param);

		// Result
		Mat result;

		matchTemplate(image_for_search, search_template, result, TM_CCORR_NORMED);
		normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		disparities.at<uchar>(v_max-pad, u-pad) = (result.cols-1) - maxLoc.x;
	}

	for(int v = v_max-1; v > pad-1; v--) {
		for(int u = pad; u < u_max + 1; u++) {
			std::set<int> search_range;
			if(u-pad-1 >= 0){
				int retrieved_d = disparities.at<uchar>(v-pad+1, u-pad-1);
				for(int i = retrieved_d-tau; i <= retrieved_d + tau; i++) {
					search_range.insert(i);
				}
			}

			int retrieved_d = disparities.at<uchar>(v-pad+1, u-pad);
			for(int i = retrieved_d-tau; i <= retrieved_d + tau; i++) {
					search_range.insert(i);
			}

			if(u-pad+1 < cols) {
				int retrieved_d = disparities.at<uchar>(v-pad+1, u-pad+1);
				for(int i = retrieved_d-tau; i <= retrieved_d + tau; i++) {
					search_range.insert(i);
				}
			}

			// Template
			Rect temp_param(u-pad, top, w, w);
			Mat search_template(padded_img_r, temp_param);

			int highest_correlation = -1;
			int highest_correlation_disparity = 0;
			
			for(auto d = search_range.begin(); d != search_range.end(); ++d){
				int left = u-(*d)-pad;
				if(left >= 0 && left + w < padded_cols) {
					// Image for search
					Rect img_param(left, v-pad, w, w);
					Mat image_for_search(padded_img_r, img_param);

					// Result
					Mat result;
					matchTemplate(image_for_search, search_template, result, TM_CCORR_NORMED);
					int value = result.at<uchar>(0,0);
					if(value > highest_correlation) {
						highest_correlation = value;
						highest_correlation_disparity = *d;
					}
				}
			}
			disparities.at<uchar>(v-pad, u-pad) = highest_correlation_disparity;
		}
	}
	imwrite("disparities.jpg", disparities);
}

Mat read_image(const string filepath) {
	Mat image = imread(filepath, 0);
	if(image.empty()) {
		cout << "Could not read image: " << filepath << endl;
		exit(1);
	}
	return image;
}

int main(int argc, char** argv )
{
	if ( argc != 7 ) {
		cout << "Usage: " << argv[0] << "<input left image> "
				<< "<input right image> <paper> <w> <dmax> <tau>" << endl << endl;
		return 0;
	}

	const string img_l_fp(argv[1]);
	const string img_r_fp(argv[2]);
	const string paper(argv[3]);
	const int w(stoi(argv[4], nullptr));
	const int dmax(stoi(argv[5], nullptr));
	const int tau(stoi(argv[6], nullptr));

	Mat img_l = read_image(img_l_fp);
	Mat img_r = read_image(img_r_fp);

	if(paper == "2013") {
		paper_2013(img_l, img_r, w, dmax, tau);
	}



	return 0;
}
