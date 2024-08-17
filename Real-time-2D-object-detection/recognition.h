/*
Arun_Madhusudhanan
Project_3 spring 2023
This is the header file for the recognition.cpp file. This file contains the function declarations for the functions in the recognition.cpp file.
*/


#include <opencv2/opencv.hpp>

/* This function is used to apply thresholding on the image
*  @param src: input image
*  @param dst: output image
*/
int thresholding(cv::Mat &src,cv::Mat &dst);

/*This function is used to apply erosion on the image. 
* @param src: input image
* @param dst: output image
* @param erosion_elem: type of structuring element
* @param erosion_size: factor to control the size of structuring element. size = 2*erosion_size + 1
* @param erosion_iter: number of iterations
*/
int erosion(cv::Mat &src,cv::Mat &dst,int erosion_elem,int erosion_size, int erosion_iter);

/* This function is used to apply dilation on the image. 
* @param src: input image
* @param dst: output image
* @param dilation_elem: type of structuring element
* @param dilation_size: factor to control size of structuring element. size = 2*dilation_size + 1
* @param dilation_iter: number of iterations
*/
int dilation(cv::Mat &src,cv::Mat &dst,int dilation_elem,int dilation_size,int erosion_iter);

/*This function is used to apply connected component analysis on the image. Uses a union find algorithm to find the connected components.(coded from scratch, extension 1)
* @param src: input image
* @param reg: output image
*/
int connectedcomponnect(cv::Mat &src,cv::Mat &reg);

/* This function is used to find the connected components in the image using in built function
@param src: input image
@param reg: output image with regions identified
@param labelimage: image with region labels marked
@param valid_labels: vector of valid labels. Regions with pixels less than min_pixel_count are not considered as valid labels
@param centroids: centroids of the regions
*/
void connectedcomponent(cv::Mat &src,cv::Mat &reg,cv::Mat &labelimage,std::vector<int> &valid_labels,cv::Mat &centroids);

/*This function gets the class name of the object from the label entered by user.
if the label is not present in the map, it asks the user to enter the name of the object to be added in the map
* @param c: label of the object
* @return: name of the object
*/
std::string classname(char c,std::map<char, std::string> &label_map);

/* This function finds the nearest neighbour of the target feature vector in the database. Uses the scaled Euclidean distance as distance metric
* @param labels: labels of the database images extrcted from csv
* @param database_features: feature vectors of the database images
* @param target_feature: feature vector of the target image
*/
int nearneighbour(std::vector<char *> &labels,std::vector<std::vector<float>> &database_features,std::vector<float> &target_feature,char* label);

/* This function finds the k nearest neighbour of the target feature vector in the database. Uses the scaled Euclidean distance as distance metric.
* @param labels: labels of the database images extracted from csv
* @param database_features: feature vectors of the database images
* @param target_feature: feature vector of the target image
* @param label: label of the nearest neighbour
* @param k: number of nearest neighbours to consider
*/
int KN_nearest(std::vector<char *> &labels, std::vector<std::vector<float>> &database_features, std::vector<float> &target_feature, char* label, int k);