//Arun_Madhusudhanan
//Project_1 spring 2023
//imgDisplay.cpp read an image file and display it in a window. If the user types 'q', the program will quit. 


#include <iostream>
#include <opencv2/opencv.hpp>

// argc is the number of command line arguments
// argv is an array of character arrays (command line arguments)
// argv[0] is the name of the executable function
int main(int argc, char *argv[]){
cv::Mat src; //allocates input image
char filename[256];

std::cout<<"This program read an image file and display it in a window. If the user types 'q', the program will quit.\n";
if (argc < 2){
        printf("Usage is %s image_filename\n", argv[0]);
        exit(-1);
    }

else strcpy(filename, argv[1]); // copy command line filename to a local variable
src = cv::imread(filename); // reads the image from a file, allocates space for it

if (src.empty()) { // check if imread was successful
        printf("Failed to read image file\n");
        exit(-2);
    }

//create a window
cv::namedWindow( "imgdisplay",cv::WINDOW_KEEPRATIO);

// show the image in a window
cv::imshow("imgdisplay",src);


while(true){ 
    // wait for a key press  
    char ch = cv::waitKey(0);
    if (ch=='q') break; //If the user types 'q', the program will quit.   
} 

//close window
cv::destroyWindow("imgdisplay"); 
return(0);
}