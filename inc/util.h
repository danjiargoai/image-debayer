#pragma once

#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// take number image type number (from cv::Mat.type()), get OpenCV's enum string.
string getImgType(int imgTypeInt);

void debayer_u8xbpp_hqlinear(const cv::Mat& in, cv::Mat& out, int offset_x, int offset_y);

void generate_filtered_image(const Mat& input_img, Mat& img_ll, Mat& img_lh, Mat& img_hl, Mat& img_hh, Vec3f& low_filter, Vec3f& high_filter);

// Generate a bayer raw image. 
void generate_bayer_raw(const Mat& img_in, Mat& img_out, Mat& img_r, Mat& img_g, Mat& img_b, bayer_pattern_e pattern);

float calc_mse(const Mat& img1, const Mat& img2);

#include "util.inl"
