/*
Arun Madhusudhanan
Project 4  spring 2023
This program is used to project virtual objects on the target. Targets used are chessboard and aruco markers
Press 't' to project a trapezoid on the chessboard ( Note: this is only implemented for chessboard)
Press 'c' to project a chair on the target ( Note: this is only implemented for chessboard)
Press 'l' to project a l-shape on the target (chessborad or aruco marker)
Press 'o' to superimpose a picture on the chessboard ( Note: this is only implemented for chessboard)
Press 'q' to quit the program
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
  

//verify if video channel is open
if(!cap->isOpened()){
std::cout<<"Not able open the camera device.\n";
exit(-1);
}

char filename[256]; //yaml file name which contains camera matrix and distortion coefficients
char image_name[256]; //used for storing image name for overlay
if (argc < 2)
    {
        std::cout << "Please enter the  calibration data yaml file name" << std::endl;
        printf("usage: %s <filename>\n", argv[0]);
        exit(-1);
    }
else
    {
        strcpy(filename, argv[1]);
    }

cv::Size chesspatternsize(9,6); //interior number of chesscorners (width, height)
std::vector<cv::Point2f> chesscorners; //this will be filled by the detected chesscorners
cv::Mat camera_matrix_chess, distortion_coeff_chess; //camera matrix and distortion coefficients
camera_matrix_chess = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); //initialize camera matrix to all zeros

std::vector<cv::Vec3f> chessworldCoordinates; //world coordinates of the chessboard corners
std::vector<std::vector<cv::Vec3f> > chesspoint_list; //list of world coordinates of the chessboard corners
std::vector<std::vector<cv::Point2f> > chesscorner_list; //list of image coordinates of the chessboard corners

cv::Mat rvec_chess, tvec_chess; //rotation and translation vectors for chess board

cv::Size arucopatternsize(7,5); //interior number of arucocorners (width, height)
std::vector<cv::Point2f> arucocorners; //this will be filled by the detected chesscorners
cv::Mat camera_matrix_aruco, distortion_coeff_aruco;
// std::vector<cv::Mat> rot_aruco, trans_aruco;
camera_matrix_aruco = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); //initialize camera matrix for aruco to all zeros

std::vector <cv::Vec3f> arucoworldCoordinates; //world coordinates of the aruco corners
std::vector<std::vector<cv::Vec3f> > arucopoint_list; //list of world coordinates of the aruco corners
std::vector<std::vector<cv::Point2f> > arucocorner_list; //list of image coordinates of the aruco corners

char yaml_file[256]; //used for storing yaml file 
// Read the camera matrix and distortion coefficients to a file
strcpy(yaml_file, filename); 
// strcat(yaml_file, "/");
// strcat(yaml_file, "calibration_data.yml");

// load the camera matrix and distortion coefficients from a file
cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
fs["camera_matrix"] >> camera_matrix_chess;
fs["distortion_coefficients"] >> distortion_coeff_chess;

fs["camera_matrix"] >> camera_matrix_aruco;
fs["distortion_coefficients"] >> distortion_coeff_aruco;
fs.release();

// construct world coordinates for the chessboard chesscorners
WorldCoordinates(chessworldCoordinates, chesspatternsize, 1.0);

// construct world coordinates for the aruco corners
aruco_WorldCoordinates(arucoworldCoordinates, arucopatternsize);

//identifies a display window
cv::namedWindow("Video_Display",cv::WINDOW_KEEPRATIO);
// cv::namedWindow("Video_Display",1);

//initialize the parameter for superimposing an image
bool superimpose = false;

//initialize the parameter for projecting a trapezoid
bool is_trapezoid = false;

//initialize the parameter for projecting a l-shape
bool is_lshape = false;

//initialize the parameter for projecting a chair
bool is_chair = false;

cv::Mat frame; //current frame

//loop for capturing, manipulating and displaying frames
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
    
    //program quit if the user types 'q'
    if (key=='q') break;  
    
    //superimpose an image on the chessboard if user types 'o'
    else if (key == 'o') {
        superimpose = true;
        is_trapezoid = false;
        is_lshape = false;
        is_chair = false;
        if (argc < 3){
        std::cout << "Please enter the file location of the image to be overlayed" << std::endl;
        printf("usage: %s <filename> <imagename>\n", argv[0]);
        exit(-1);
        }
        else
        {
        strcpy(image_name, argv[2]); //copy the image name to the image_name variable
        }        
    }

    //cancel superimposing an image on the chessboard if user types 'p'    
    else if (key == 'f') {
        superimpose = false;
        is_trapezoid = false;
        is_lshape = false;
        is_chair = false;
    }  

    //project a trapezoid on the chessboard if user types 't'
    else if (key == 't') {
        superimpose = false;
        is_trapezoid = true;
        is_lshape = false;
        is_chair = false;
    }  
    //project a L shape on the chessboard if user types 'l'
    else if (key == 'l') {
        superimpose = false;
        is_trapezoid = false;
        is_lshape = true;
        is_chair = false;
    }

    else if (key == 'c') {
        superimpose = false;
        is_trapezoid = false;
        is_lshape = false;
        is_chair = true;
    }           

    // extract chessboard chesscorners
    bool chesscorners_found = getChessboardCorners(frame, chesspatternsize, chesscorners);

    // cv::Mat img = cv::imread("/home/arun/PRCV/project_4/worldcup.jpg");

    cv::Mat img = cv::imread(image_name);

    
    std::vector<cv::Point2f> pts_tgt; // store the destination points, i.e. the corners of the target

    if (chesscorners_found) {       
        drawChessboardCorners(frame, chesspatternsize, chesscorners, chesscorners_found);
        //find the list of world coordinates of the chessboard corners
        chesspoint_list.push_back(chessworldCoordinates);
        //find the outer corners of the chessboard    
        chesscorner_list.push_back(chesscorners);
        // Use solvePnP to find the rotation and translation vectors        
        bool check_chesscorners = cv::solvePnP(chessworldCoordinates, chesscorners, camera_matrix_chess, distortion_coeff_chess, rvec_chess, tvec_chess); 
        //uncomment the following lines to print the rotation and translation vectors
        // std::cout <<"rotation vector: " << rvec_chess << std::endl;
        // std::cout <<"translation vector: " << tvec_chess << std::endl;
        if (check_chesscorners){
            //draw the coordinate axes
            outsidecorners(frame, chessworldCoordinates, rvec_chess, tvec_chess, camera_matrix_chess, distortion_coeff_chess);
            //project virtual objects on the chessboard
            if (is_trapezoid)  trapezoid(frame, rvec_chess, tvec_chess, camera_matrix_chess, distortion_coeff_chess); //project a trapezoid on the chessboard
            if (is_chair) chair(frame, rvec_chess, tvec_chess, camera_matrix_chess, distortion_coeff_chess);
            if (is_lshape) l_shape(frame, rvec_chess, tvec_chess, camera_matrix_chess, distortion_coeff_chess); //project a L shape on the chessboard
            // superimpose an image on the chessboard
            if (superimpose) {                
                // corner indices of the chessboard
                int corner_index[] = {0, 8, 45, 53};
                for (int i : corner_index) {            
                    pts_tgt.push_back(chesscorners[i]);            
                }
                // Image size
                int width = img.size().width;
                int height = img.size().height;            
                std::vector<cv::Point2f> pts_img; // store the source points, i.e. the corners of the image
                pts_img.push_back(cv::Point2f(0, 0));
                pts_img.push_back(cv::Point2f(width, 0));
                pts_img.push_back(cv::Point2f(0, height));
                pts_img.push_back(cv::Point2f(width, height));
                // Compute homography from source and destination points
                cv::Mat h = cv::findHomography(pts_img, pts_tgt);
                cv::Mat img_out;
                // Warp source image to destination based on homography
                cv::warpPerspective(img, img_out, h, frame.size());
                for (int i = 0; i < frame.rows; i++) {
                    for (int j = 0; j < frame.cols; j++) {
                        if (img_out.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
                            frame.at<cv::Vec3b>(i, j) = img_out.at<cv::Vec3b>(i, j);
                        }
                    }
                }         
            }
        }

        
    }   
    
    // project a virtual image on the aruco board if detected
    std::vector<int> aruco_markerIds; //vector to store the ids of the detected markers
    std::vector<std::vector<cv::Point2f> > aruco_markerCorners; //vector to store the corners of the detected markers
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); //create a dictionary for the markers
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);// create a board object
    cv::aruco::detectMarkers(frame, dictionary, aruco_markerCorners, aruco_markerIds); //detect the markers
    //if markers are detected
    if(aruco_markerIds.size()>0){
        cv::aruco::drawDetectedMarkers(frame, aruco_markerCorners, aruco_markerIds); //draw the markers
        cv::Mat rvec_aruco, tvec_aruco;
        int valid =cv::aruco::estimatePoseBoard(aruco_markerCorners, aruco_markerIds, board, camera_matrix_aruco, distortion_coeff_aruco, rvec_aruco, tvec_aruco); //estimate the pose of the board
        if (valid){
            // trapezoid(frame, rvec_aruco, tvec_aruco, camera_matrix_aruco, distortion_coeff_aruco);
            if (is_trapezoid) std::cout << "Pyramid virtual object can only be projected on the chessboard, press 'l' to draw l shape on aruco board" << std::endl;
            if (is_chair) std::cout << "Chair virtual object can only be projected on the chessboard, press 'l' to draw l shape on aruco board" << std::endl;
            else if (is_lshape) aruco_l_shape(frame, rvec_aruco, tvec_aruco, camera_matrix_aruco, distortion_coeff_aruco);
            //draw the coordinate axes
            cv::drawFrameAxes(frame, camera_matrix_aruco, distortion_coeff_aruco, rvec_aruco, tvec_aruco, 0.1);
        }
    }
    
    // Display the resulting frame
    cv::imshow("Video_Display", frame);
}
cv::destroyWindow("Video_Display");
return(0);

}