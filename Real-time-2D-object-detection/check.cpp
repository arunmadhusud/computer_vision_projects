/*
Arun_Madhusudhanan
Project_3 spring 2023
This code is for 2D object recognition. 
usage: ./objectrec <directory path> <csv filename> classifier_type operation_mode
It identify a specified set of objects placed on a white surface in a translation, scale, and rotation invariant manner from a camera looking straight down. 
if 't' is used as operation mode , system will go on training mode, where it will ask for the label of the object and store the features of the object in a csv file.
if 'c' is used as operation mode, system will go on classification mode, where it will detect the object and display the class of the object.
Two different types of classififer are used. Use 'n' as classifier_type is for nearest neighbour and 'k' as classifier_type  for k-nearest neighbour.
*/

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "recognition.h"
#include "csv_util.h"


int main(int argc, char *argv[]){

cv::Mat src; //allocates input image
cv::Mat dst; //allocates output image
char dirname[256]; //directory of image database
char buffer[256]; //buffer for building class names
char filename[256]; //csv filename to store features
char feature_type[256]; //store the feature type from command line  
FILE *fp;
DIR *dirp;
struct dirent *dp; 
int image_count = 0; 
char classification_type[256]; //store the classification type from command line,n for nearest neighbour and k for k-nearest neighbour
char operation_type[256]; //store the operation mode from command line,t for training mode and c for classification mode

//allocates the map for storing the labels and class names
std::map<char, std::string> label_map {
            {'p', "pen"}, {'l', "lid"}, {'z', "zed"}, {'g', "glasscase"},
             {'c', "calculator"}, {'b', "beeropener"},
            {'m', "mask"}, {'i', "packet clip"},{'o', "cap"},{'s', "spoon"},{'k', "cloth"},
            {'w', "wallet"},{'a', "airpodcase"} , {'v', "watch"} 
};

// check for sufficient arguments
if( argc < 5) {
printf("usage: %s <directory path> <csv filename> classifier_type operation_mode\n", argv[0]);
exit(-1);
}

// get the directory path
strcpy(dirname, argv[1]);
printf("Processing directory %s\n", dirname );
std::cout<<std::endl;


// open the directory
dirp = opendir( dirname );
if( dirp == NULL) {
printf("Cannot open directory %s\n", dirname);
exit(-1);
}


// csv file for storing features
strcpy(filename, argv[2]);

// get the classification type, n for nearest neighbour and k for k-nearest neighbour
strcpy(classification_type, argv[3]);

// get the operation mode, t for training mode and c for classification mode
strcpy(operation_type, argv[4]);

bool training;
if (strcmp(operation_type, "t") == 0) {
    training = true; // flag to switch between training and classification mode
    std::cout << "Training Mode is ON" << std::endl;
}
else if (strcmp(operation_type, "c") == 0) {
    training = false;
    std::cout << "Classification Mode is ON" << std::endl;
}
else {
    std::cout << "Invalid operation mode" << std::endl;
    exit(-1);
}



while( (dp = readdir(dirp)) != NULL ) {
    
    // check if the file is an image
    if( strstr(dp->d_name, ".jpeg") ||
    strstr(dp->d_name, ".png") ||
    strstr(dp->d_name, ".ppm") ||
    strstr(dp->d_name, ".tif") ) {

    printf("processing image file: %s\n", dp->d_name);
    std::cout<<std::endl;

    // build the overall filename
    strcpy(buffer, dirname);
    strcat(buffer, "/");
    strcat(buffer, dp->d_name);

    printf("full path name: %s\n", buffer);
    std::cout<<std::endl;
    src = cv::imread(buffer);

    cv::namedWindow( "imgdisplay",cv::WINDOW_KEEPRATIO); //create a window to display the image

    cv::Mat input;
    if(training==false){
    src.copyTo(input);
    cv::namedWindow( "Input_image",cv::WINDOW_KEEPRATIO); //create a window to display the image
    // show the image in a window
    cv::imshow("Input_image",input); //display the image 
    }     
      
    
    cv::Mat threshold; //allocates threshold image
    thresholding(src,threshold); // pre-processing and thresholding
    // this works with 2 iteraions of my thresholding funcion, erosion function and dilation function. 

    cv::Mat dilated;
    dilation(threshold,dilated,0,1,3); // dilation function to clean up the image. 3 iterations are required.
    //dilate(threshold, dilated);

    cv::Mat eroded;
    erosion(dilated,eroded,1,1,3); // erosion function to clean up the image. 3 iterations are required.
    //erode(dilated,eroded);
    

    cv::Mat reg(eroded.size(), CV_8UC3); //allocates region image
    int regioncount = 0;
    // cv::Mat pass(eroded.size(), CV_8UC3); //allocates region image
    // connectedcomponnect(eroded,pass);
    // cv::namedWindow( "connected_component",cv::WINDOW_KEEPRATIO);        
    //     // show the image in a window
    // cv::imshow("connected_component",pass);
    std::vector<int> valid_labels; //vector to store valid regions. only regions with pixel count greater than a threshold are considered valid
    cv::Mat labelimage(src.size(), CV_32S);
    cv::Mat centroids;
    connectedcomponent(eroded,reg,labelimage,valid_labels,centroids); // connected component function to segment the regions in the image
    
/*Based on user imput, the program either goes into training mode or classification mode.
In training mode, the user is asked to enter the class label for each region in the image if the label is not already present in the label map in getclasslabel function.
In classification mode, the program classifies each region in the image based on the nearest neighbour or k-nearest neighbour algorithm.
*/    
    cv::Moments moment; // moments of the region
    cv::Point2f center; // center of the region
    double h_wratio; // height to width ratio of the bounding box of the region
    double percent_fill; // percentage of the region that is filled in the bounding box
    std::vector<float> feature; // vector to store the features of the region(moment, center, height to width ratio, percentage of the region that is filled in the bounding box)
    std::vector<float> nu22_angle(valid_labels.size(),0);// vector to store the moments around the central axis of the region
    
    for (int k = 0; k < valid_labels.size(); k++) {
        feature.clear();
        nu22_angle.clear();
        int curr_label = valid_labels[k]; // current label
        cv::Mat curr_region; // allocates current region
        curr_region = (labelimage == curr_label);   // get the current region 
        moment = cv::moments(curr_region,true); // get the moments of the region
        center = cv::Point2f(moment.m10/moment.m00, moment.m01/moment.m00);  // get the center of the region         
        double angle = 0.5* atan2((2 * moment.nu11 ),(moment.nu20 - moment.nu02)); // get the angle of the central axis of the region
        double beta = angle + 0.5*CV_PI; // get the beta angle to find the moments around the central axis of the region
        int x_max = INT_MIN, x_min = INT_MAX, y_max = INT_MIN, y_min = INT_MAX; // variables to find the bounding box of the region
        int temp_1, temp_2; // variables to find the bounding box of the region
        int x_project,y_project,width,height; // variables to find the bounding box of the region        
        for (int i = 0; i < curr_region.rows; i++) {
            for (int j = 0; j < curr_region.cols; j++) {
                if (curr_region.at<uchar>(i, j) == 255) {
                    x_project = (i - center.x ) * cos(angle) + (j - center.x ) * sin(angle); // project the points of the region on the central axis
                    y_project = -(i - center.x) * sin(angle) + (j - center.x ) * cos(angle); // project the points of the region on the central axis
                    x_max = std::max(x_max, x_project); 
                    x_min = std::min(x_min, x_project);
                    y_max = std::max(y_max, y_project);
                    y_min = std::min(y_min, y_project);                    
                    nu22_angle[k] = nu22_angle[k] + (((j - center.y ) * cos(beta) + (i - center.x) * sin(beta))*((j - center.y ) * cos(beta) + (i - center.x) * sin(beta)))/moment.m00;// get the moments around the central axis of the region
                }
            }
        }
        // Log scale moment
        nu22_angle[k] = -1 * copysign(1.0, nu22_angle[k]) * log10(abs(nu22_angle[k])); // convert the moments around the central axis of the region to log scale
        width = x_max - x_min; // get the width of the bounding box of the region
        height = y_max - y_min; // get the height of the bounding box of the region
        // swap the width and height if the width is less than the height
        if(width < height){
            temp_1 = width;        
            width = height;        
            height = temp_1;                  
        }        
        
        cv::Size rect_size = cv::Size(width, height); // get the size of the bounding box of the region
        cv::RotatedRect rect = cv::RotatedRect(center,rect_size,angle* 180.0 / CV_PI);  // get the bounding box of the region  

        h_wratio = (double) height/width; // get the height to width ratio of the bounding box of the region
        percent_fill = moment.m00/(width*height);  // get the percentage of the region that is filled in the bounding box  
        double axis_length = width/2; // get the length of the central axis of the region
        double l1 = axis_length * sin(angle); 
        double l2 = sqrt(axis_length * axis_length - l1 * l1);
        double x_p = center.x + l2, y_p = center.y + l1;
        // cv::arrowedLine(src, center, cv::Point2f(x_p, y_p), cv::Scalar(0, 0, 255), 2, cv::LINE_8);  // draw the central axis of the region  

        cv::Point2f vertices[4]; // get the vertices of the bounding box of the region
        rect.points(vertices);
        // for (int i = 0; i < 4; i++) {
        //     line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 3); // draw the bounding box of the region
        // }

        
        cv::namedWindow( "imgdisplay",cv::WINDOW_KEEPRATIO);
        
        // show the image in a window
        cv::imshow("imgdisplay",src);
        
        //find the hu moments of the region        
        //https://learnopencv.com/shape-matching-using-hu-moments-c-python/
        double huMoments[7]; 
        cv::HuMoments(moment, huMoments);
        // Log scale hu moments 
        for(int i = 0; i < 4; i++) {
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])); // convert the hu moments to log scale
            feature.push_back(huMoments[i]); // add the hu moments to the feature vector
        }

        // std::cout << "Hu Moments: " << huMoments[0] << " " << huMoments[1] << " " << huMoments[2] << " " << huMoments[3] << " " << huMoments[4] << " " << huMoments[5] << " " << huMoments[6] << std::endl;
        // std::cout <<"nu22_angle: " << nu22_angle[k] << std::endl;    
        // std::cout <<"h_wratio: " << h_wratio << std::endl;    
        // std::cout <<"percent_fill: " << percent_fill << std::endl;

        // feature.push_back(nu22_angle[k]); // add the moments around the central axis of the region to the feature vector
        feature.push_back(h_wratio); // add the height to width ratio of the bounding box of the region to the feature vector
        feature.push_back(percent_fill); // add the percentage of the region that is filled in the bounding box to the feature vector

        //https://www.geeksforgeeks.org/write-on-an-image-using-opencv-in-cpp/    
        std::string ratio = std::to_string(float(h_wratio));
        std::string text = "h_wratio: " + ratio;
        // cv::putText(src,text,center,cv::FONT_HERSHEY_SIMPLEX , 0.5 ,cv::Scalar(0, 0, 255), 2, cv::LINE_AA); // add the height to width ratio of the bounding box of the region to the image
        
         
        if(training){
            
            // cv::arrowedLine(reg, center, cv::Point2f(x_p, y_p), cv::Scalar(0, 0, 255), 3, cv::LINE_8);  // draw the central axis of the region
            // for (int i = 0; i < 4; i++) {
            //     line(curr_region, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 3); // draw the bounding box of the region
            // } 
            // cv::putText(reg,text,center,cv::FONT_HERSHEY_COMPLEX , 0.5 ,cv::Scalar(0, 120 , 255), 1, -1);             
            // cv::namedWindow( "thresholded image",cv::WINDOW_KEEPRATIO); 
            // // show the image in a window
            // cv::imshow("thresholded image",threshold);
            // //create a window
            // cv::namedWindow( "dilated image",cv::WINDOW_KEEPRATIO);
            // // show the image in a window
            // cv::imshow("dilated image",dilated);
            // //create a window
            // cv::namedWindow( "eroded image",cv::WINDOW_KEEPRATIO);
            // // show the image in a window
            // cv::imshow("eroded image",eroded);
            // cv::namedWindow( "regions",cv::WINDOW_KEEPRATIO);
            // // show the image in a window
            // cv::imshow("regions",reg);

            cv::namedWindow("Current Region", cv::WINDOW_KEEPRATIO); //create a window to display the current region
            cv::imshow("Current Region", curr_region);

            std::cout << "Do you want to label this region?" << std::endl; //if the image is being trained, then ask the user if they want to label the region
            std::cout << "Press 'y' to label this region." << std::endl;
            std::cout << "Press 'n' to skip this region." << std::endl;
            std::cout << "Press 'q' to quit" << std::endl;
            std::cout<<std::endl;
            std::cout <<"Current Labels: " << std::endl;
            for (auto it = label_map.begin(); it != label_map.end(); ++it) {
                std::cout << it->first << ": " << it->second << std::endl;
            }
            std::cout<<std::endl;
            char key = cv::waitKey(0);
            if(key == 'y'){
                std::cout << "Input the class for this object." << std::endl;  //if the user wants to label the region, then ask the user what class the region belongs to
                char k1 = cv::waitKey(0);
                std::string className = classname(k1,label_map ); //get the class name based on the key pressed. if label is not found, then program will ask whether to add the label to the label file          
                char* label = new char[className.length() + 1];
                strcpy(label, className.c_str()); //convert the class name to a char array
                append_image_data_csv(filename, label, feature, 0); //append the class name and feature vector to the csv file
                image_count++;
            }
            else if (key == 'n'){
                std::cout << "Skipping this region." << std::endl;  //if the user inputs an invalid key, then skip the region
            }
            else if (key=='q'){
                std::cout << "Quitting." << std::endl; //if the user inputs q, then quit the program
                return -1;
            }
            // else{
            //     std::cout << "Invalid input. Skipping this region." << std::endl; //if the user inputs an invalid key, then skip the region
            // }
            cv::destroyWindow("Current Region");
        }        
        else{  

            cv::arrowedLine(src, center, cv::Point2f(x_p, y_p), cv::Scalar(0, 0, 255), 2, cv::LINE_8);  // draw the central axis of the region             
            for (int i = 0; i < 4; i++) {
                line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 3); // draw the bounding box of the region
            }            
        
            std::cout << "Classifying this region." << std::endl; //if the image is not being trained, then classify the region
            std::vector<char *> labels; //filenames of images read from csv file
            std::vector<std::vector<float>> data; //features of images read from csv file
            std::vector<float> distance; //distance between traget image and database image
            char label[256] = {};
            read_image_data_csv(filename,labels, data, 0); 
            // scaled_euclidean_distance(feature, data,distance);
            if (strcmp(classification_type, "n") == 0){
                nearneighbour(labels, data, feature, label); //classify the region using the nearest neighbour algorithm
            }
            else if (strcmp(classification_type, "k") == 0){
                int k =3;
                KN_nearest(labels, data, feature, label,k); //classify the region using the k nearest neighbour algorithm
            }
            else{
                std::cout << "Invalid input for classification type" << std::endl; //if the user inputs an invalid key, then skip the region
                return -2;               
            }
            

            cv::putText(src,label,center,cv::FONT_HERSHEY_SIMPLEX , 2 ,cv::Scalar(0, 0, 255), 3, cv::LINE_AA); //add the class name to the image
            cv::imshow("imgdisplay",src);

            /* Extension -2
            if the object is not trained, then switch to tarining mode 
            */
            if(strcmp(label,"unknown")==0) {
                cv::namedWindow("Current Region", cv::WINDOW_KEEPRATIO); //create a window to display the current region
                cv::imshow("Current Region", curr_region);
                std::cout << "Do you want to label this region?" << std::endl; //if the image is being trained, then ask the user if they want to label the region
                std::cout << "Press 'y' to label this region." << std::endl;
                std::cout << "Press 'n' to skip this region." << std::endl;
                std::cout << "Press 'q' to quit" << std::endl;
                std::cout<<std::endl;
                std::cout <<"Current Labels: " << std::endl;
                for (auto it = label_map.begin(); it != label_map.end(); ++it) {
                    std::cout << it->first << ": " << it->second << std::endl;
                }
                std::cout<<std::endl;
                char key = cv::waitKey(0);
                if(key == 'y'){
                    std::cout << "Input the class for this object." << std::endl;  //if the user wants to label the region, then ask the user what class the region belongs to
                    char k1 = cv::waitKey(0);
                    std::string className = classname(k1,label_map ); //get the class name based on the key pressed. if label is not found, then program will ask whether to add the label to the label file          
                    char* label = new char[className.length() + 1];
                    strcpy(label, className.c_str()); //convert the class name to a char array
                    append_image_data_csv(filename, label, feature, 0); //append the class name and feature vector to the csv file
                    image_count++;
                }
                else if (key == 'n'){
                    std::cout << "Skipping this region." << std::endl;  //if the user inputs an invalid key, then skip the region
                }
                else if (key=='q'){
                    std::cout << "Quitting." << std::endl; //if the user inputs q, then quit the program
                    return -1;
                }               
                cv::destroyWindow("Current Region");
            }
            std::cout << std::endl;
            std::cout <<"Do you want to continue classifying?" << std::endl; //ask the user if they want to continue classifying
            std::cout << "Press 'y' to continue classifying." << std::endl;
            std::cout << "Press 'q' to quit." << std::endl;
            char key = cv::waitKey(0);
            if(key == 'y'){
                std::cout << "Continuing." << std::endl;
            }
            else if (key == 'q'){
                std::cout << "Quitting." << std::endl;
                return -3;
            }
            else{
                std::cout << "Invalid input. Continuing." << std::endl;                
            }
            
        }
           
    }    
    
    cv::namedWindow( "imgdisplay",cv::WINDOW_KEEPRATIO); //create a window to display the image

    // show the image in a window
    cv::imshow("imgdisplay",src); //display the image   
    std::cout << std::endl;    
   
    cv::destroyAllWindows(); 
    }
}
return(0);

}