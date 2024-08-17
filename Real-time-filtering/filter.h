//Arun_Madhusudhanan
//Project_1 spring 2023
//functionsfor custom effects
//this is the include file

#include <opencv2/opencv.hpp>
//funtion to Display alternative greyscale live video
int greyscale( cv::Mat &src, cv::Mat &dst );
//funtion to Implement a 5x5 Gaussian filter as separable 1x5 filters.
int blur5x5( cv::Mat &src, cv::Mat &dst );
// funtion to implement Sobel X as sperable 1*3 filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
// funtion to implement Sobel Y as sperable 1*3 filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
//function that generates a gradient magnitude image from the X and Y Sobel images.
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
//function that blurs and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
// function to Implement a live video cartoonization function using the gradient magnitude and blur/quantize filters.
int cartoon( cv::Mat &src, cv::Mat &dst, int levels, int magThreshold );
//function to Pixelate an image
int pixelate(cv::Mat& src, cv::Mat& dst, int pixel_size);
//function to adjust brightness and contrast of an image
int brightness_contrast(cv::Mat &src, cv::Mat &dst,double alpha,int beta);