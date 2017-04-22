#include <string>
using namespace std;

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

// High Quality Linear algorithm
void debayer_u8xbpp_hqlinear(const cv::Mat& in, cv::Mat& dest, int offset_x, int offset_y)
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

	for (int row = 2; row < height - 2; ++row) {
		for (int col = 2; col < width - 2; ++col) {
			// Debayer algorithm starts here
			float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
			s1 = in.at<uchar>(row, col-2) + in.at<uchar>(row, col+2);
			s2 = in.at<uchar>(row-2, col) + in.at<uchar>(row+2, col);
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

void generate_filtered_image(const Mat& input_img, Mat& img_ll, Mat& img_lh, Mat& img_hl, Mat& img_hh, Vec3f& low_filter, Vec3f& high_filter) {
  sepFilter2D(input_img, img_ll, -1, low_filter, low_filter);
  sepFilter2D(input_img, img_lh, -1, low_filter, high_filter);
  sepFilter2D(input_img, img_hl, -1, high_filter, low_filter);
  sepFilter2D(input_img, img_hh, -1, high_filter, high_filter);
}

// Generate a bayer raw image. 
void generate_bayer_raw(const Mat& img_in, Mat& img_out, Mat& img_r, Mat& img_g, Mat& img_b, bayer_pattern_e pattern) {
  // Initialize bayer patterm, R, G, B
  size_t num_rows = img_in.rows;
  size_t num_cols = img_in.cols;
  size_t half_rows = num_rows / 2 + (num_rows % 2 == 0 ? 0 : 1);
  size_t half_cols = num_cols / 2 + (num_cols % 2 == 0 ? 0 : 1);
  img_out.create(num_rows, num_cols, CV_8U);
  img_r.create(half_rows, half_cols, CV_8U);
  img_g.create(num_rows, half_cols, CV_8U);
  img_b.create(half_rows, half_cols, CV_8U);

  bool even_row = false;
  bool even_col = false;
  switch(pattern) {
    case BGGR:
      for (int row = 0; row < num_rows; ++row) {
        even_col = false;
        for (int col = 0; col < num_cols; ++col) {
          Vec3b pix = img_in.at<Vec3b>(row, col);
          if (even_row ^ even_col) {
            img_out.at<uchar>(row, col) = pix.val[1];
            img_g.at<uchar>(row, col/2) = pix.val[1];
          }
          else if (even_row && even_col) {
            img_out.at<uchar>(row, col) = pix.val[2];
            img_r.at<uchar>(row/2, col/2) = pix.val[2];
          }
          else {
            img_out.at<uchar>(row, col) = pix.val[0];
            img_b.at<uchar>(row/2, col/2) = pix.val[0];
          }
          even_col = !even_col;
        }
        even_row = !even_row;
      }
      break;
    case RGGB:
      for (int row = 0; row < num_rows; ++row) {
        even_col = false;
        for (int col = 0; col < num_cols; ++col) {
          Vec3b pix = img_in.at<Vec3b>(row, col);
          if (even_row ^ even_col) {
            img_out.at<uchar>(row, col) = pix.val[1];
            img_g.at<uchar>(row, col/2) = pix.val[1];
          }
          else if (even_row && even_col) {
            img_out.at<uchar>(row, col) = pix.val[0];
            img_b.at<uchar>(row/2, col/2) = pix.val[0];
          }
          else {
            img_out.at<uchar>(row, col) = pix.val[2];
            img_r.at<uchar>(row/2, col/2) = pix.val[2];
          }
          even_col = !even_col;
        }
        even_row = !even_row;
      }
      
      break;
  }
}

// Calculate Mean Square Error
float calc_mse(const cv::Mat& img1, const cv::Mat& img2) {
  assert(img1.rows == img2.rows);
  assert(img1.cols == img2.cols);
  cv::Mat diff = img1 - img2;
  cv::Mat square_diff;
  cv::pow(diff, 2, square_diff);
  float total_square_diff = 0.0f;
  for (int ch = 0; ch < img1.channels(); ++ch) {
    total_square_diff += cv::sum(square_diff)[ch];
  }
  return total_square_diff / (img1.rows * img1.cols * img1.channels());
}
