/*   
    Arun Madhusudhanan
    Project 4 spring 2023
    This program demonstrates the use of the Harris corner detector
    Press 'h' to enable the Harris corner detector
    Press 'r' to reset the Harris corner detector
    Press 'q' to quit the program
*/

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "definitions.h"

int main(int argc, char *argv[]){
//Open webcam video channel
cv::VideoCapture *cap;
cap = new cv::VideoCapture(0);

// get some properties of the image
cv::Size refS( (int) cap->get(cv::CAP_PROP_FRAME_WIDTH ),
(int) cap->get(cv::CAP_PROP_FRAME_HEIGHT));  

//verify if video channel is open
if(!cap->isOpened()){
std::cout<<"Not able oprn the camera device.\n";
exit(-1);
}

//identifies a display window
cv::namedWindow("Video_Display",cv::WINDOW_KEEPRATIO);

cv::Mat frame; //current frame

//Harris corner detector parameters
int blocksize = 2;
int aperture = 3;
double k = 0.04;

//flag to enable/disable Harris corner detector
bool is_harris = false;

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

    //check for user input
    char key = cv::waitKey(10);    

    //if user presses 'q' then exit the loop
    if(key == 'q'){
        break;
    }
    //if user presses 'h' then enable/disable the Harris corner detector
    else if (key == 'h'){
        is_harris = true;
    }
    //if user presses 'r' then reset the Harris corner detector
    else if (key == 'r'){
        is_harris = false;
    }
    cv::Mat gray;
    if (is_harris){
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //convert to grayscale
        cv::Mat dst = cv::Mat::zeros(gray.size(), CV_32FC1); //create a matrix to store the Harris corner detector response
        cv::cornerHarris(gray, dst, blocksize, aperture, k, cv::BORDER_DEFAULT);   //apply the Harris corner detector    
        double max,min; //variables to store the maximum and minimum values of the Harris corner detector response
        cv::minMaxLoc(dst, &min, &max);  //find the maximum and minimum values of the Harris corner detector response       
        for(int i = 0; i < dst.rows; i++){
            float *rptr = dst.ptr<float>(i);
            for(int j = 0; j < dst.cols; j++){ 
                //if the Harris corner detector response is greater than 15% of the maximum response, then draw a circle at that corner               
                if(rptr[j] > 0.15*max){
                    cv::circle(frame, cv::Point(j,i), 5, cv::Scalar(0,0,255), 2,8,0); //draw a circle at the corner
                }
            }
        }        
    }
    //display the frame
    cv::imshow("Video_Display", frame);
}
//release the video capture object
cv::destroyWindow("Video_Display");
return(0);
}