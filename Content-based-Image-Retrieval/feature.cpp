/*
Arun_Madhusudhanan
Project_2 spring 2023
This code extracts the features of traget image, opens the features of image in database stored in csv file and compute the distance between features.
Then it display top N matches
User inputs: feature type, image database path, csv filename
Usage: ./<project_name> <target_image_filename> feature_type matching_method num_matching_images <database_feature_filename> 
*/


#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "matching.h"
#include "csv_util.h"


int main(int argc, char *argv[]){
cv::Mat src; //allocates input image
cv::Mat dst; //allocates output image for display
std::vector<float> feature_target_image; //allocates a vector for features of target image
char feature_type[256]; //store the feature type from command line 
char match_type[256]; //store the distance metric type from command line 

char filename[256];
if (argc < 6){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename \n", argv[0]);
        exit(-1);
    }
else strcpy(filename, argv[1]); // copy command line filename to a local variable for target image
src = cv::imread(filename); // reads the image from a file, allocates space for it

// method for extracting features
strcpy(feature_type, argv[2]);

// method for extracting features
strcpy(match_type, argv[3]);

std::vector<char *> filenames; //filenames of images read from csv file
std::vector<std::vector<float>> data; //features of images read from csv file
std::vector<float> distance; //distance between traget image and database image

/*Extract features based on user input and calculate the distance between features based on user input */

/* "b" : Use baseline matching (intensity values of middle 9*9 pixels) as features.
         Use sum of squared difference ("sq") as distance metric for better results*/
if (strcmp(feature_type, "b") == 0){    
    baseline_match(src,feature_target_image);  
    char file[256]; //filename of csv file where the baseline matching features of database images are stored
    strcpy(file, argv[5]);       
    read_image_data_csv(file,filenames, data, 0); 
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data);      
}

/*"hm" is to use single normalized color histogram for whole image as features.  
   Use histogram instersection ("hi") as distance metric for better results.*/ 
if (strcmp(feature_type, "hm") == 0){
    int bins = 32;    
    histogram_match(src,bins,feature_target_image); 
    char file[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file, argv[5]);         
    read_image_data_csv(file,filenames, data, 0); 
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data); 
    std::cout << "done";
}

/*"mhm": Use mutihistogram matching. Feature 1: single normalized color histogram for middle 200*200 pixels 
         Feature 2: use single normalized color histogram for whole image as features. Use histogram instersection ("hi") as distance metric for better results.
*/
if (strcmp(feature_type, "mhm") == 0){    
    int bins = 32;    
    histogram_match(src,bins,feature_target_image);  
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data); 
    
    std::vector<float> feature_target_image_ci;
    centre_image(src,bins,feature_target_image_ci); 
    char file2[256]; //filename of csv file where the single normalized color histogram features of centre portion of images are stored
    strcpy(file2, argv[6]);
    std::vector<char *> filenames2; //filenames of images read from csv file
    std::vector<std::vector<float>> data2; //features of images read from csv file
    std::vector<float> distance2; //distance between traget image and database image    
    read_image_data_csv(file2,filenames2, data2, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image_ci,distance2, data2);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image_ci,distance2, data2);

    for (int i=0;i<distance.size();i++)  {
        distance[i] = 0.6 * distance[i] + 0.4 * distance2[i];
       
    }     
     
}

/*"ts": Use texture (based on sobel magnitude and orientation) and color histogram for matching. 
        Use histogram instersection ("hi") as distance metric  for better results.*/
if (strcmp(feature_type, "ts") == 0){
    int bins = 32;    
    histogram_match(src,bins,feature_target_image); 
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data);
    
    std::vector<float> feature_target_image_ts; //allocates a vector for features based on whole image texture histogram
    int texture_bins = 8;
    texture(src,texture_bins,feature_target_image_ts);
    char file2[256]; //filename of csv file where the single normalized whole image texture histogram features of database images are stored
    strcpy(file2, argv[6]);
    std::vector<char *> filenames2; //filenames of images read from csv file
    std::vector<std::vector<float>> data2; //features of images read from csv file
    std::vector<float> distance2; //distance between traget image and database image       
    read_image_data_csv(file2,filenames2, data2, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image_ts,distance2, data2);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image_ts,distance2, data2); 

    for (int i=0;i<distance.size();i++)  {
        distance[i] = 0.5 * distance[i] + 0.5 * distance2[i];       
    }
     
}

/*
 "tl": Use texture (based on laws filters) and color histogram for matching. Use histogram instersection ("hi") as distance metric for better results.   
*/
if (strcmp(feature_type, "tl") == 0){
    int bins = 32;    
    histogram_match(src,bins,feature_target_image); 
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data); 
    
    std::vector<float> feature_target_image_tl;
    int laws_bins = 8;
    laws(src,laws_bins,feature_target_image_tl);
    char file2[256]; //filename of csv file where the single normalized whole image texture features of database images are stored
    strcpy(file2, argv[6]);
    std::vector<char *> filenames2; //filenames of images read from csv file
    std::vector<std::vector<float>> data2; //features of images read from csv file
    std::vector<float> distance2; //distance between traget image and database image       
    read_image_data_csv(file2,filenames2, data2, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image_tl,distance2, data2);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image_tl,distance2, data2);
    for (int i=0;i<distance.size();i++)  {
        distance[i] = 0.5 * distance[i] + 0.5 * distance2[i];       
    } 
     
}

/*
"gf": Use texture (based on gabor filters) and color histogram for matching. Use histogram instersection ("hi") as distance metric for better results.
*/

if (strcmp(feature_type, "gf") == 0){
    int bins = 32;    
    histogram_match(src,bins,feature_target_image); 
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    std::cout << feature_target_image.size()<<"\n";
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data); 
    
    std::vector<float> feature_target_image_gf; 
    int gabor_bins = 8;   
    gaborfilter(src,gabor_bins,feature_target_image_gf);
    std::cout << feature_target_image_gf.size()<<"\n";
    char file2[256]; //filename of csv file where the single normalized whole image texture  features of database images are stored
    strcpy(file2, argv[6]);
    std::vector<char *> filenames2; //filenames of images read from csv file
    std::vector<std::vector<float>> data2; //features of images read from csv file
    std::vector<float> distance2; //distance between traget image and database image       
    read_image_data_csv(file2,filenames2, data2, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image_gf,distance2, data2);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image_gf,distance2, data2);
    for (int i=0;i<distance.size();i++)  {
        distance[i] = 0.5 * distance[i] + 0.5 * distance2[i];       
    } 
     
}

/*
"mct":Use texture (based on sobel magnitude and orientation) and color histogram of centre portion for matching. Use histogram instersection ("hi") as distance metric for better results.
*/
if (strcmp(feature_type, "mct") == 0){
    int bins = 8;    
    midcolortext(src,bins,feature_target_image); 
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the  features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data);  
     
     
}

/*
"sv": Use spatial variance of color and  color histogram. Use histogram instersection ("hi") as distance metric  for better results
*/
if (strcmp(feature_type, "sv") == 0){
    int bins = 32;    
    histogram_match(src,bins,feature_target_image); 
    if (argc < 7){
        printf("Usage is %s target_image_filename feature_type matching_method num_matching_images database_feature_filename1 database_feature_filename2 \n", argv[0]);
        exit(-1);
    }    
    char file1[256]; //filename of csv file where the single normalized color histogram features of database images are stored
    strcpy(file1, argv[5]);       
    read_image_data_csv(file1,filenames, data, 0);
    if (strcmp(match_type, "sq") == 0) sumofsqdiff(feature_target_image,distance, data);
    if (strcmp(match_type, "hi") == 0) hm_inter(feature_target_image,distance, data); 
    
    std::vector<float> feature_target_image_sv;      
    spacialVariance(src,feature_target_image_sv);
    char file2[256]; //filename of csv file where the features based on spatial variance of color of database images are stored
    strcpy(file2, argv[6]);
    std::vector<char *> filenames2; //filenames of images read from csv file
    std::vector<std::vector<float>> data2; //features of images read from csv file
    std::vector<float> distance2; //distance between traget image and database image       
    read_image_data_csv(file2,filenames2, data2, 0);    
    specialVariance_distance(feature_target_image_sv,distance2, data2);    
    for (int i=0;i<distance.size();i++)  {
        distance[i] = 0.3 * distance[i] + 0.7 * distance2[i];       
    } 
     
}

//fSort filenames based on the increasing order of distance
pairsort(distance, filenames);
std::cout << std::endl;
int match_num = std::stoi(argv[4]);

//display the filenames of top matches in command line and display the top matches in a window
for (int i=0;i<match_num;i++){
    std::cout<<(filenames[i])<<"\n";
    dst = cv::imread(filenames[i]);
    std::string window_name = "imgdisplay_" + std::to_string(i);
    cv::namedWindow( window_name,cv::WINDOW_KEEPRATIO);
    cv::imshow(window_name,dst);     
}

//destroy the window on a keyboard press
cv::waitKey(0);
for (int i=0;i<match_num;i++){
    std::string window_name = "imgdisplay_" + std::to_string(i);
    cv::destroyWindow(window_name); 
}

}

