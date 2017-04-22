#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include "defines.h"
#include "util.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
  if ( argc != 2 )
  {
    printf("usage: ColorPlaneInterpolation <Image_Path>\n");
    return -1;
  }

  // Read in the raw image
  Mat image_orig;
  image_orig = imread( argv[1], 1 );

  if ( !image_orig.data )
  {
     printf("No image data \n");
     return -1;
  }

  size_t half_rows = image_orig.rows / 2 + (image_orig.rows % 2 == 0 ? 0 : 1);
  size_t half_cols = image_orig.cols / 2 + (image_orig.cols % 2 == 0 ? 0 : 1);

  // Initialize bayer, R, G, B
  Mat image_bayer, image_r, image_g, image_b;
  generate_bayer_raw(image_orig, image_bayer, image_r, image_g, image_b, RGGB);

  // Debayer using High Quality Linear Algorithm 
  Mat image_debayered_hqlinear;
  debayer_u8xbpp_hqlinear(image_bayer, image_debayered_hqlinear, 0, 0);
  float mse_hqlinear = calc_mse(image_orig, image_debayered_hqlinear);

  Vec3f low_pass_filter(0.25f, 0.5f, 0.25f);
  Vec3f high_pass_filter(0.25f, -0.5f, 0.25f);

  Mat image_ll, image_hh, image_lh, image_hl;
  generate_filtered_image(image_orig, image_ll, image_lh, image_hl, image_hh, low_pass_filter, high_pass_filter);

#ifdef DEBUG
  cout << "High Qiality Linear algorithm MSE:   " << mse_hqlinear << endl;
  imwrite("output/bayer.jpg", image_bayer);
  imwrite("output/R.jpg", image_r);
  imwrite("output/G.jpg", image_g);
  imwrite("output/B.jpg", image_b);
  imwrite("output/hq_bayer.jpg", image_debayered_hqlinear);
  imwrite("output/ll.jpg", image_ll);
  Mat image_lh_norm;
  normalize(image_lh, image_lh_norm, 0, 255, NORM_MINMAX, CV_8UC3);
  imwrite("output/lh.jpg", image_lh_norm);
  Mat image_hl_norm;
  normalize(image_hl, image_hl_norm, 0, 255, NORM_MINMAX, CV_8UC3);
  imwrite("output/hl.jpg", image_hl_norm);
  Mat image_hh_norm;
  normalize(image_hh, image_hh_norm, 0, 255, NORM_MINMAX, CV_8UC3);
  imwrite("output/hh.jpg", image_hh_norm);
  imwrite("output/orig.jpg", image_orig);
#endif
  waitKey(0);

  return 0;
}
