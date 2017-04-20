#pragma once

#include <string>
#include <opencv2/opencv.hpp>
using namespace std;

// take number image type number (from cv::Mat.type()), get OpenCV's enum string.
string getImgType(int imgTypeInt);

void debayer_rggb_to_u8xbpp_hqlinear(const cv::Mat& in, cv::Mat& out, int offset_x, int offset_y);