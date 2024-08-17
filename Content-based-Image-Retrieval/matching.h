//Arun_Madhusudhanan
//Project_2 spring 2023
//decleration of library of functions for feature exraction and distance calculation.



#include <opencv2/opencv.hpp>

// Given an image (src), use the 9x9 square in the middle of the image as a feature vector for baseline matching
int baseline_match(cv::Mat &src, std::vector<float> &features);

//function to sort filenames based on the increasing order of distance
int pairsort(std::vector<float> &distance, std::vector<char *> &filenames);

/*Given a RGB image, this function gives a single normalized rg chromaticity histogram as the
feature vector. No of bins for the histogram has to be passed as an argument.*/
int histogram_match(cv::Mat &src,int &bins, std::vector<float> &features);

//calculate sum-of-squared-difference of features for the distance metric
int sumofsqdiff(std::vector<float> &feature,std::vector<float> &distance, std::vector<std::vector<float>> data);

//calculate histogram instersection of features for the distance metric
int hm_inter(std::vector<float> &feature,std::vector<float> &distance, std::vector<std::vector<float>> data);

/*Given a RGB image, this function gives a single normalized rg chromaticity histogram of centre
200 x 200 pixels as the feature vector. No of bins for the histogram has to be passed as an argument.
*/
int centre_image(cv::Mat &src,int &bins, std::vector<float> &features);

/* This function gnerates a whole image texture histogram as the feature vector
Given a RGB image, this function gives a single normalized 2D histogram of gradient orientation and magnitude as the
feature vector.  No of bins for the histogram has to be passed as an argument.
*/
int texture(cv::Mat &src, int &bins, std::vector<float> &features);

/*Given an RGB image, this function applies a combination of law's filters 
on image and extract the 1D histogram responses as texture feature. Combinations 
of (l5,e5),(e5,s5),(s5,w5),(r5,w5) law's filters are used. Pixel values of each 
filtered image is used to create a 1D histogram.
*/
int laws(cv::Mat &src, int &bins,std::vector<float> &features);

/*Given an RGB image, this function applies a 48 gabor filters 
on image and extract the 1D histogram responses as texture feature.
48 gabor filters were obtained by changing sigma values ( no = 3) and 
orientations (no = 16). Pixel values of each filtered image is used to create a 
1D histogram.
*/
int gaborfilter(cv::Mat &src,int &bins,std::vector<float> &features);

/*Task 5 - Approach 1
Given an RGB image, this function computes 2D color histogram of the
middle part of image and add that as feature. Then it computes 2D texture
histogram based on sobel magnitude and orientation and add it to the feature. */
int midcolortext(cv::Mat &src,int &bins,std::vector<float> &features);

/*Task 5 - Approach 2
Given an RGB image, this function computes the spatial variance of color corresponding 
to each RGB bin and store it as feature. Used standard deviation as spatial variance metric.
*/
int spacialVariance(cv::Mat &src, std::vector<float> &features);

/*Distance metric for spatial variance.used histogram intersection method. */
int specialVariance_distance(std::vector<float> &features,std::vector<float> &distance, std::vector<std::vector<float>> data);
