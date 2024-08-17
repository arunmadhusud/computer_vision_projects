/*
Arun Madhusudhanan
Project 4  spring 2023
This program is used to calibrate the camera using chessboard images
Press 's' to save the current frame for calibration, program need atleast 5 images for calibration
The images used for calibration are stored in the folder "foldername"
Press 'c' to calibrate the camera
The camera matrix and distortion coefficients are saved to a file "calibration_data.yml" in the folder "foldername"
Press 'q' to quit the program
usage : ./calib <foldername>
*/


#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "definitions.h"
#include <opencv2/aruco.hpp>


int main(int argc, char *argv[]){

//Open webcam video channel
cv::VideoCapture *cap;
cap = new cv::VideoCapture(0);

//uncomment following to calibrate phone camera
// cap = new cv::VideoCapture("http://10.110.131.247:8080/video");

char foldername[256]; //used for storing image names
if (argc < 2)
    {
        std::cout << "Please enter the folder name where the calibration images and calibration yaml files are stored" << std::endl;
        exit(-1);
    }
else
    {
        strcpy(foldername, argv[1]);
    }


// get some properties of the image
cv::Size refS( (int) cap->get(cv::CAP_PROP_FRAME_WIDTH ),
(int) cap->get(cv::CAP_PROP_FRAME_HEIGHT));  

//verify if video channel is open
if(!cap->isOpened()){
std::cout<<"Not able open the camera device.\n";
exit(-1);
}


cv::Size chesspatternsize(9,6); //interior number of chesscorners (width, height)
std::vector<cv::Point2f> chesscorners; //this will be filled by the detected corners
cv::Mat camera_matrix_chess, distortion_coeff_chess; //camera matrix and distortion coefficients
std::vector<cv::Mat> rot_chess, trans_chess; //rotation and translation vectors

std::vector<cv::Vec3f> chessworldCoordinates; //world coordinates of the chessboard corners
std::vector<std::vector<cv::Vec3f> > chesspoint_list; //list of world coordinates of the chessboard corners
std::vector<std::vector<cv::Point2f> > chesscorner_list; //list of image coordinates of the chessboard corners

char buffer[256]; //used for storing image names
char yaml_file[256]; //used for storing yaml file name

// construct world coordinates for the chessboard corners
WorldCoordinates(chessworldCoordinates, chesspatternsize, 1.0);

//identifies a display window
cv::namedWindow("Video_Display",cv::WINDOW_KEEPRATIO);

cv::Mat frame; //current frame
//loop for capturing, manipulating and displaying frames

int frame_cout = 0; //counts the number of frames captured
while(true){
    //capture frame     
    *cap>> frame;

    //check if frame is empty   
    if(frame.empty()){
        std::cout<<"frame is empty.\n";
        break;
    }

    // change the size of the frame
    resize(frame, frame, cv::Size(), 0.5, 0.5);
    
    // make a copy of the frame
    cv::Mat frame_copy = frame.clone();

    //check for user input
    char key = cv::waitKey(10);           

    // detect extract chessboard corners
    bool chesscorners_found = getChessboardCorners(frame, chesspatternsize, chesscorners);

    // draw the chessboard corners
    if (chesscorners_found) {
        drawChessboardCorners(frame, chesspatternsize, chesscorners, chesscorners_found);
    }
    // initialize the camera matrix and distortion coefficients
   

    if (frame_cout == 0){
        std::cout<< "Initializing camera matrix and distortion coefficients" << std::endl;
        // float initial_camera_matrix[3][3] = {{1, 0, float(frame.cols)/2}, {0, 1, float(frame.rows)/2}, {0, 0, 1}};        
        camera_matrix_chess = cv::Mat::eye(3, 3, CV_64FC1); 
        camera_matrix_chess.at<double>(0,2) = float(frame.cols)/2;
        camera_matrix_chess.at<double>(1,2) = float(frame.rows)/2;
        distortion_coeff_chess = cv::Mat::zeros(1, 5, CV_64FC1);  
        std::cout<< "Camera matrix before calibration: " << camera_matrix_chess << std::endl;
        std::cout<< "Distortion Coefficients before calibration: " << distortion_coeff_chess << std::endl;
        std::cout<< std::endl;
        frame_cout++;
    }   

    // if the user types 's', select calibration images and store them
    if (key=='s') {
        if (chesscorners_found){
            std::cout << "corners detected" << std::endl;
            // store the world coordinates of the chessboard corners
            chesspoint_list.push_back(chessworldCoordinates);
            // store the image coordinates of the chessboard corners
            chesscorner_list.push_back(chesscorners);         
            // calculate the camera matrix and distortion coefficients using the stored chessboard corners      
            double error = calibrateCamera(chesspoint_list, chesscorner_list, frame.size(), camera_matrix_chess, distortion_coeff_chess, rot_chess, trans_chess);
            std::cout << "Camera matrix before final calibration: " << camera_matrix_chess << std::endl;            
            std::cout<< std::endl;     
            std::cout << "Distortion Coefficients before final calibration: " << distortion_coeff_chess << std::endl;            
            std::cout<< std::endl;
            // store the images themselves that are being used for a calibration                        
            // strcpy(buffer, "/home/arun/PRCV/project_4"); //change this to argv[1] if you want to use the command line argument later
            // strcat(buffer, "/");
            // strcat(buffer, "calibration_images");
            strcpy(buffer, foldername);
            strcat(buffer, "/");
            strcat(buffer, "calibration_image_");
            strcat(buffer, std::to_string(chesspoint_list.size()).c_str());
            strcat(buffer, ".png");
            std::cout << "saving calibration image " << buffer << std::endl;
            // save the image
            cv::imwrite(buffer,frame);                    
        }
        else {
            std::cout << "corners not detected" << std::endl;
        }
    }

    // if the user types 'c', calibrate the camera and display the camera matrix
    else if (key=='c') {
        if (chesspoint_list.size() > 5) {
            // calibrate the camera                       
            double error = calibrateCamera(chesspoint_list, chesscorner_list, frame.size(), camera_matrix_chess, distortion_coeff_chess, rot_chess, trans_chess, cv::CALIB_FIX_ASPECT_RATIO);
            // display the reprojection error
            std::cout << "Final Re-projection Error: " << error << std::endl;
            // display the camera matrix and distortion coefficients
            std::cout << "Final camera matrix: " << camera_matrix_chess << std::endl;                
            std::cout << "Final distortion Coefficients: " << distortion_coeff_chess << std::endl;             
            // save the camera matrix and distortion coefficients to a file           
            strcpy(yaml_file, foldername);
            strcat(yaml_file, "/");
            strcat(yaml_file, "calibration_data.yml");
            cv::FileStorage fs(yaml_file, cv::FileStorage::WRITE); 
            if (fs.isOpened()) {
                // file opened successfully, write data to file
                fs << "camera_matrix" << camera_matrix_chess;
                fs << "distortion_coefficients" << distortion_coeff_chess;
                fs.release();
            } 
            else {
                // file failed to open for writing
                std::cout << "Failed to open file for writing." << std::endl;
            }       
        }
        else {
            std::cout << "not enough calibration images, atleast 5 images required" << std::endl;
        }
    }
    //program quit if the user types 'q'
    else if (key=='q') break;
    //display the frame
    cv::imshow("Video_Display",frame);
        
}
//release the capture device
cv::destroyWindow("Video_Display");
return(0);
}