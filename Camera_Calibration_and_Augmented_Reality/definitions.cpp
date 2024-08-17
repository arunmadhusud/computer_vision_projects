/*Arun Madhusudhanan
Project_4 for spring 2023
library of functions for the project
*/

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types_c.h>
#include "definitions.h"

/* 
    * This function is used for detecting the corners of the chessboard and extracting the corners
    * @parameter frame: the current frame of the video
    * @parameter patternsize: the size of the chessboard 
    * @parameter corners: the corners of the chessboard in the current frame
*/
bool getChessboardCorners(cv::Mat &frame, cv::Size patternsize, std::vector<cv::Point2f> &corners) {
    cv::Mat gray;
    cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
    bool corners_found = findChessboardCorners(gray, patternsize, corners); // find the corners   
    if (corners_found) {        
        cv::Size subPixWinSize(10, 10); // the size of the window for cornerSubPix
        cv::TermCriteria termCrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 1, 0.1); // the termination criteria for cornerSubPix        
        cornerSubPix(gray, corners, subPixWinSize, cv::Size(-1, -1), termCrit); // refine the corners
        }
    return corners_found;
}


/*
    * This function is used to construct world coordinates for the chessboard corners
    * @parameter worldCoordinates: the vector of world coordinates
    * @parameter patternsize: the size of the chessboard
    * @parameter squaresize: the size of the squares on the chessboard
*/
void WorldCoordinates(std::vector<cv::Vec3f> &worldCoordinates, cv::Size patternsize, float squaresize) {
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            worldCoordinates.push_back(cv::Vec3f(j * squaresize, -i * squaresize, 0)); // the z coordinate is 0 because the chessboard is on the plane z = 0            
        }
    }
}

/*
    * This function is used to construct world coordinates for the aruco corners
    * @parameter worldCoordinates: the vector of world coordinates
    * @parameter patternsize: the size of the aruco
*/
void aruco_WorldCoordinates(std::vector<cv::Vec3f> &worldCoordinates, cv::Size patternsize){
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            worldCoordinates.push_back(cv::Vec3f(i*0.05 , j*0.05 , 0)); // the z coordinate is 0 because the chessboard is on the plane z = 0           
        }
    }
}

/*
    * This function is used to display the matrix in the console
    * @parameter matrix: the matrix to be displayed
*/
void disp_matrix(cv::Mat &matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            std::cout << matrix.at<double>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

/*
    * This function is used to draw the 3D axes on the origin of the chessboard
    * @parameter frame: the current frame
    * @parameter points: the world coordinates of the chessboard corners
    * @parameter rvec: the rotation vector
    * @parameter tvec: the translation vector
    * @parameter camera_matrix: the camera matrix
    * @parameter distortion_coeff: the distortion coefficients
*/
void outsidecorners(cv::Mat &frame, std::vector<cv::Vec3f> points, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff) {
    std::vector<cv::Point2f> imagePoints;
    projectPoints(points, rvec, tvec, camera_matrix, distortion_coeff, imagePoints); // project the world coordinates to image coordinates
    // uncomment the following lines to draw cirlces on the corners
    int corner_index[] = {0, 8, 45, 53};
    for (int i : corner_index) {
        circle(frame, imagePoints[i], 6, cv::Scalar(0, 255, 0), 4);
    }
    cv::Rodrigues(rvec, rvec); // convert rotation vector to rotation matrix
    cv::drawFrameAxes(frame, camera_matrix, distortion_coeff, rvec, tvec, 3); // draw the axes
}


/*
This function is used to create a trapezoid as virtual object
    * @param frame: the current frame
    * @param rvec: the rotation vector
    * @param tvec: the translation vector
    * @param camera_matrix: the camera matrix
    * @param distortion_coeff: the distortion coefficients
*/
void trapezoid(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff) {    
    // define the world coordinates of the trapezoid
    std::vector<cv::Vec3f> objectPoints;
    objectPoints.push_back(cv::Vec3f(2, -1, 0));
    objectPoints.push_back(cv::Vec3f(2, -4, 0));
    objectPoints.push_back(cv::Vec3f(5, -1, 0));
    objectPoints.push_back(cv::Vec3f(5, -4, 0));
    objectPoints.push_back(cv::Vec3f(2.5, -2, 1));
    objectPoints.push_back(cv::Vec3f(2.5, -3, 1));
    objectPoints.push_back(cv::Vec3f(4.5, -2, 1));
    objectPoints.push_back(cv::Vec3f(4.5, -3, 1));
    // project the world coordinates to image coordinates
    std::vector<cv::Point2f> imagePoints;
    projectPoints(objectPoints, rvec, tvec, camera_matrix, distortion_coeff, imagePoints);
    // draw the trapezoid    
    cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[3], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(0, 0, 255), 2);    
    cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[7], cv::Scalar(0, 0, 255), 2);
}

/*
This function is used to create a L shape as virtual object
    * @param frame: the current frame
    * @param rvec: the rotation vector
    * @param tvec: the translation vector
    * @param camera_matrix: the camera matrix
    * @param distortion_coeff: the distortion coefficients
*/

void l_shape(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff) {  
    // define the world coordinates of the L shape
    std::vector<cv::Vec3f> objectPoints;    
    objectPoints.push_back(cv::Vec3f(2, -1, 0));
    objectPoints.push_back(cv::Vec3f(3, -1, 0));
    objectPoints.push_back(cv::Vec3f(3, -4, 0));
    objectPoints.push_back(cv::Vec3f(5, -4, 0));
    objectPoints.push_back(cv::Vec3f(5, -5, 0));
    objectPoints.push_back(cv::Vec3f(2, -5, 0));
    objectPoints.push_back(cv::Vec3f(2, -1, 1));
    objectPoints.push_back(cv::Vec3f(3, -1, 1));
    objectPoints.push_back(cv::Vec3f(3, -4, 1));
    objectPoints.push_back(cv::Vec3f(5, -4, 1));
    objectPoints.push_back(cv::Vec3f(5, -5, 1));
    objectPoints.push_back(cv::Vec3f(2, -5, 1));
    // project the world coordinates to image coordinates
    std::vector<cv::Point2f> imagePoints;
    projectPoints(objectPoints, rvec, tvec, camera_matrix, distortion_coeff, imagePoints);
    // draw the L shape
    cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[4], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[0], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[7], imagePoints[8], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[8], imagePoints[9], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[9], imagePoints[10],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[10],imagePoints[11],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[11],imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[0], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[8], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[9], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[10],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[11],cv::Scalar(0, 0, 255), 2); 
}

/*
This function is used to create a L shape as virtual object on the aruco marker
   * @param frame: the current frame
   * @param rvec: the rotation vector
   * @param tvec: the translation vector
   * @param camera_matrix: the camera matrix
   * @param distortion_coeff: the distortion coefficients
*/
void aruco_l_shape(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff) {  
    // define the world coordinates of the L shape
    std::vector<cv::Vec3f> objectPoints;      
    objectPoints.push_back(cv::Vec3f(0, 0.1, 0));
    objectPoints.push_back(cv::Vec3f(0, 0.15, 0));
    objectPoints.push_back(cv::Vec3f(0.2, 0.15, 0));
    objectPoints.push_back(cv::Vec3f(0.2, 0.25, 0));
    objectPoints.push_back(cv::Vec3f(0.25, 0.25, 0));
    objectPoints.push_back(cv::Vec3f(0.25, 0.1, 0));
    objectPoints.push_back(cv::Vec3f(0, 0.1, 0.05));
    objectPoints.push_back(cv::Vec3f(0, 0.15, 0.05));
    objectPoints.push_back(cv::Vec3f(0.2, 0.15, 0.05));
    objectPoints.push_back(cv::Vec3f(0.2, 0.25, 0.05));
    objectPoints.push_back(cv::Vec3f(0.25, 0.25, 0.05));
    objectPoints.push_back(cv::Vec3f(0.25, 0.1, 0.05)); 
    // project the world coordinates to image coordinates  
    std::vector<cv::Point2f> imagePoints;
    projectPoints(objectPoints, rvec, tvec, camera_matrix, distortion_coeff, imagePoints);
    // draw the L shape
    cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[4], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[0], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[7], imagePoints[8], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[8], imagePoints[9], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[9], imagePoints[10],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[10],imagePoints[11],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[11],imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[0], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[8], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[9], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[10],cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[11],cv::Scalar(0, 0, 255), 2);    
}


void chair(cv::Mat &frame, cv::Mat rvec, cv::Mat tvec, cv::Mat camera_matrix, cv::Mat distortion_coeff){
    std::vector<cv::Vec3f> objectPoints;
    objectPoints.push_back(cv::Vec3f(2.8, -1.8, 0));
    objectPoints.push_back(cv::Vec3f(2.8, -2.2, 0));
    objectPoints.push_back(cv::Vec3f(3.2, -2.2, 0));
    objectPoints.push_back(cv::Vec3f(3.2, -1.8, 0));
    objectPoints.push_back(cv::Vec3f(2.8, -1.8, 1.5));
    objectPoints.push_back(cv::Vec3f(2.8, -2.2, 1.5));
    objectPoints.push_back(cv::Vec3f(3.2, -2.2, 1.5));
    objectPoints.push_back(cv::Vec3f(3.2, -1.8, 1.5));    

    objectPoints.push_back(cv::Vec3f(4.8, -1.8, 0));
    objectPoints.push_back(cv::Vec3f(4.8, -2.2, 0));
    objectPoints.push_back(cv::Vec3f(5.2, -2.2, 0));
    objectPoints.push_back(cv::Vec3f(5.2, -1.8, 0));
    objectPoints.push_back(cv::Vec3f(4.8, -1.8, 1.5));
    objectPoints.push_back(cv::Vec3f(4.8, -2.2, 1.5));
    objectPoints.push_back(cv::Vec3f(5.2, -2.2, 1.5));
    objectPoints.push_back(cv::Vec3f(5.2, -1.8, 1.5));  

    objectPoints.push_back(cv::Vec3f(2.8, -3.8, 0));
    objectPoints.push_back(cv::Vec3f(2.8, -4.2, 0));
    objectPoints.push_back(cv::Vec3f(3.2, -4.2, 0));
    objectPoints.push_back(cv::Vec3f(3.2, -3.8, 0));
    objectPoints.push_back(cv::Vec3f(2.8, -3.8, 1.5));
    objectPoints.push_back(cv::Vec3f(2.8, -4.2, 1.5));
    objectPoints.push_back(cv::Vec3f(3.2, -4.2, 1.5));
    objectPoints.push_back(cv::Vec3f(3.2, -3.8, 1.5));

    objectPoints.push_back(cv::Vec3f(4.8, -3.8, 0));
    objectPoints.push_back(cv::Vec3f(4.8, -4.2, 0));
    objectPoints.push_back(cv::Vec3f(5.2, -4.2, 0));
    objectPoints.push_back(cv::Vec3f(5.2, -3.8, 0));
    objectPoints.push_back(cv::Vec3f(4.8, -3.8, 1.5));
    objectPoints.push_back(cv::Vec3f(4.8, -4.2, 1.5));
    objectPoints.push_back(cv::Vec3f(5.2, -4.2, 1.5));
    objectPoints.push_back(cv::Vec3f(5.2, -3.8, 1.5));

    objectPoints.push_back(cv::Vec3f(2.8, -1.8, 4));
    objectPoints.push_back(cv::Vec3f(2.8, -2.2, 4));
    objectPoints.push_back(cv::Vec3f(3.2, -2.2, 4));
    objectPoints.push_back(cv::Vec3f(3.2, -1.8, 4));
    objectPoints.push_back(cv::Vec3f(4.8, -1.8, 4));
    objectPoints.push_back(cv::Vec3f(4.8, -2.2, 4));
    objectPoints.push_back(cv::Vec3f(5.2, -2.2, 4));
    objectPoints.push_back(cv::Vec3f(5.2, -1.8, 4));

    std::vector<cv::Point2f> imagePoints;
    projectPoints(objectPoints, rvec, tvec, camera_matrix, distortion_coeff, imagePoints);
    cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[0], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[5], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[7], imagePoints[4], cv::Scalar(0, 0, 255), 2);  
    cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[5], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[6], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[7], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[8], imagePoints[9], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[9], imagePoints[10], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[10], imagePoints[11], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[11], imagePoints[8], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[12], imagePoints[13], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[13], imagePoints[14], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[14], imagePoints[15], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[15], imagePoints[12], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[8], imagePoints[12], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[9], imagePoints[13], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[10], imagePoints[14], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[11], imagePoints[15], cv::Scalar(0, 0, 255), 2);  

    cv::line(frame, imagePoints[16], imagePoints[17], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[17], imagePoints[18], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[18], imagePoints[19], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[19], imagePoints[16], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[20], imagePoints[21], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[21], imagePoints[22], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[22], imagePoints[23], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[23], imagePoints[20], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[16], imagePoints[20], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[17], imagePoints[21], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[18], imagePoints[22], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[19], imagePoints[23], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[24], imagePoints[25], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[25], imagePoints[26], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[26], imagePoints[27], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[27], imagePoints[24], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[28], imagePoints[29], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[29], imagePoints[30], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[30], imagePoints[31], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[31], imagePoints[28], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[24], imagePoints[28], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[25], imagePoints[29], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[26], imagePoints[30], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[27], imagePoints[31], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[4], imagePoints[21], cv::Scalar(0, 0, 255), 2);    
    cv::line(frame, imagePoints[4], imagePoints[15], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[15], imagePoints[30], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[30], imagePoints[21], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[0], imagePoints[32], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[1], imagePoints[33], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[2], imagePoints[34], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[3], imagePoints[35], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[8], imagePoints[36], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[9], imagePoints[37], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[10], imagePoints[38], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[11], imagePoints[39], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[32], imagePoints[39], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[33], imagePoints[38], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[32], imagePoints[33], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[38], imagePoints[39], cv::Scalar(0, 0, 255), 2);

    cv::line(frame, imagePoints[4], imagePoints[26], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, imagePoints[15], imagePoints[21], cv::Scalar(0, 0, 255), 2);

}