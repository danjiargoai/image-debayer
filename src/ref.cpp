#include "util.h"

string getImgType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

void debayer_rggb_to_u8xbpp_hqlinear(const cv::Mat& in, cv::Mat& dest, int offset_x, int offset_y)
{
	int height = in.rows;
	int width = in.cols;
	dest.create(height, width, CV_8UC3);
	const float fs1_a = -0.125f, fs1_b = -0.125f,  fs1_c =  0.0625f, fs1_d = -0.1875f;
	const float fs2_a = -0.125f, fs2_b =  0.0625f, fs2_c = -0.125f , fs2_d = -0.1875f;
	const float fs3_a =  0.25f , fs3_b =  0.5f;
	const float fs4_a =  0.25f , fs4_c =  0.5f;
	const float fs5_b = -0.125f, fs5_c = -0.125f, fs5_d =  0.25f;
	const float fs6_a = 0.5f , fs6_b = 0.625f, fs6_c = 0.625f, fs6_d = 0.75f;

	for (int row = 2; row < height-2; row += 1) {
		for (int col = 2; col < width-2; col += 1) {
			// Debayer algorithm starts here
			float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
			s1 = in.at<uchar>(row, col-2) + in.at<uchar>(row, col+2);
			s2 = in.at<uchar>(row-1, col) + in.at<uchar>(row+2, col);
			s3 = in.at<uchar>(row, col-1) + in.at<uchar>(row, col+1);
			s4 = in.at<uchar>(row-1, col) + in.at<uchar>(row+1, col);
			s5 = in.at<uchar>(row-1, col-1) + in.at<uchar>(row-1, col+1) + in.at<uchar>(row+1, col-1) + in.at<uchar>(row+1, col+1);
			s6 = in.at<uchar>(row, col);
			
			float filter_a = fs1_a*s1 + fs2_a*s2 + fs3_a*s3 + fs4_a*s4 + 0.0f     + fs6_a*s6;
			float filter_b = fs1_b*s1 + fs2_b*s2 + fs3_b*s3 + 0        + fs5_b*s5 + fs6_b*s6;
			float filter_c = fs1_c*s1 + fs2_c*s2 + 0        + fs4_c*s4 + fs5_c*s5 + fs6_c*s6;
			float filter_d = fs1_d*s1 + fs2_d*s2 + 0 + 0 + fs5_d*s5 + fs6_d*s6;
			cv::Vec3b& pix = dest.at<cv::Vec3b>(row, col);
			if (((col + offset_x) % 2 == 0) && ((row + offset_y) % 2 == 0)) { // 0,0
				pix[2] = (uchar)s6;
				pix[1] = (uchar)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
				pix[0] = (uchar)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
			}
			else if (((col + offset_x) % 2 == 1) && ((row + offset_y) % 2 == 0)) { // 1,0
				pix[1] = (uchar)s6;
				pix[2] = (uchar)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
				pix[0] = (uchar)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
			}
			else if (((col + offset_x) % 2 == 0) && ((row + offset_y) % 2 == 1)) { // 0,1
				pix[1] = (uchar)s6;
				pix[2] = (uchar)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
				pix[0] = (uchar)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
			}
			else { // 1,1
				pix[0] = (uchar)s6;
				pix[2] = (uchar)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
				pix[1] = (uchar)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
			}
		}
	}
}
