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
