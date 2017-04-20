#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include "defines.h"
#include "util.h"

using namespace cv;
using namespace std;

void generate_filtered_image(const Mat& input_img, Mat& output_img, Vec3f& row_filter, Vec3f& col_filter) {
  sepFilter2D(input_img, output_img, -1, row_filter, col_filter);
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

int main(int argc, char** argv )
{
  if ( argc != 2 )
  {
    printf("usage: ColorPlaneInterpolation <Image_Path>\n");
    return -1;
  }

  // Read in the raw image
  Mat image;
  image = imread( argv[1], 1 );

  if ( !image.data )
  {
     printf("No image data \n");
     return -1;
  }

  // Show the original image
  //namedWindow("Original Image", WINDOW_AUTOSIZE );
  //imshow("Original Image", image);

  size_t half_rows = image.rows / 2 + (image.rows % 2 == 0 ? 0 : 1);
  size_t half_cols = image.cols / 2 + (image.cols % 2 == 0 ? 0 : 1);

  // Initialize bayer, R, G, B
  Mat image_bayer, image_r, image_g, image_b;

  generate_bayer_raw(image, image_bayer, image_r, image_g, image_b, RGGB);
  
  imwrite("output/bayer.jpg", image_bayer);
  imwrite("output/R.jpg", image_r);
  imwrite("output/G.jpg", image_g);
  imwrite("output/B.jpg", image_b);

#ifdef DEBUG
  for (int col = 0; col < 10; ++col) {
    cout << (int)image.at<Vec3b>(0, col)[0] << " " << (int)image.at<Vec3b>(0, col)[1] << " " << (int)image.at<Vec3b>(0, col)[2] << endl;
    cout << (int)image_bayer.at<uchar>(0, col) << endl;
  }
#endif

  Mat debayered_image;
  debayer_rggb_to_u8xbpp_hqlinear(image_bayer, debayered_image, 0, 0);
  imwrite("output/hq_bayer.jpg", debayered_image);

  // Calculate Mean Square Error
  Mat diff = image - debayered_image;
  Mat square_diff;
  pow(diff, 2, square_diff);
  double mse = (sum(square_diff)[0] + sum(square_diff)[1] + sum(square_diff)[2]) / (image.rows * image.cols * image.channels());

  Vec<float, 3> low_pass_filter(0.25f, 0.5f, 0.25f);
  Vec<float, 3> high_pass_filter(0.25f, -0.5f, 0.25f);

  //Mat low_low;
  //generate_filtered_image(image, low_low, low_pass_filter, low_pass_filter);

  waitKey(0);

  return 0;
}
