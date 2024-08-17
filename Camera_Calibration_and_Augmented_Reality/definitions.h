#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

/*
* This function is used for detecting the corners of the chessboard and extracting the corners
    * @parameter frame: the current frame of the video
    * @parameter patternsize: the size of the chessboard 
    * @parameter corners: the corners of the chessboard in the current frame 
*/
bool getChessboardCorners(cv::Mat &frame, cv::Size patternSize, std::vector<cv::Point2f> &corners);


// bool getArucoCorners(cv::Mat &frame, cv::Size patternsize, std::vector<cv::Point2f> &corners) ;

/*
    * This function is used to construct world coordinates for the chessboard corners
    * @parameter worldCoordinates: the vector of world coordinates
    * @parameter patternsize: the size of the chessboard
    * @parameter squaresize: the size of the squares on the chessboard
*/
void WorldCoordinates(std::vector<cv::Vec3f> &worldCoordinates, cv::Size patternsize, float squaresize);

/*
    * This function is used to construct world coordinates for the aruco corners
    * @parameter worldCoordinates: the vector of world coordinates
    * @parameter patternsize: the size of the aruco
*/
void aruco_WorldCoordinates(std::vector<cv::Vec3f> &worldCoordinates, cv::Size patternsize);
// void disp_matrix(cv::Mat &matrix);

/*
    * This function is used to draw the 3D axes on the origin of the chessboard
    * @parameter frame: the current frame
    * @parameter points: the world coordinates of the chessboard corners
    * @parameter rvec: the rotation vector
    * @parameter tvec: the translation vector
    * @parameter camera_matrix: the camera matrix
    * @parameter distortion_coeff: the distortion coefficient
*/
void outsidecorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff);

/*
This function is used to create a trapezoid as virtual object
    * @param frame: the current frame
    * @param rvec: the rotation vector
    * @param tvec: the translation vector
    * @param camera_matrix: the camera matrix
    * @param distortion_coeff: the distortion coefficients
*/
void trapezoid(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff);

/*
This function is used to create a L shape as virtual object
    * @param frame: the current frame
    * @param rvec: the rotation vector
    * @param tvec: the translation vector
    * @param camera_matrix: the camera matrix
    * @param distortion_coeff: the distortion coefficients
*/
void l_shape(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff);

/*
This function is used to create a L shape as virtual object on the aruco marker
   * @param frame: the current frame
   * @param rvec: the rotation vector
   * @param tvec: the translation vector
   * @param camera_matrix: the camera matrix
   * @param distortion_coeff: the distortion coefficients
*/
void aruco_l_shape(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff) ;
// void aruco_outsidecorners(cv::Mat &frame);
// void outsidecorners_new(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff,std::vector<cv::Point2f> &pts_dst) ;

void chair(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff);
