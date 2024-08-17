/*
Arun_Madhusudhanan
Project_2 spring 2023
This code extracts the features of images in database and saves the features in a csv file.
User inputs: feature type, image database path, csv filename
Usage: ./<project_name> feature_type <directory path> <csv filename> 
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "matching.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
  cv::Mat src; //allocates input image  
  std::vector<float> features; //allocates a feature vector
  char dirname[256]; //directory of image database
  char buffer[256];
  char filename[256]; //csv filename to store features
  char feature_type[256]; //store the feature type from command line  
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;  
  int image_count = 0;

  // check for sufficient arguments
  if( argc < 4) {
    printf("usage: %s feature_type <directory path> <csv filename>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[2]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // csv file for storing features
  strcpy(filename, argv[3]);

  // method for extracting features
  strcpy(feature_type, argv[1]);
  

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
    strstr(dp->d_name, ".png") ||
    strstr(dp->d_name, ".ppm") ||
    strstr(dp->d_name, ".tif") ) {

    printf("processing image file: %s\n", dp->d_name);

    // build the overall filename
    strcpy(buffer, dirname);
    strcat(buffer, "/");
    strcat(buffer, dp->d_name);

    printf("full path name: %s\n", buffer);
    src = cv::imread(buffer); // reads the image from a file, allocates space for it
    
    /*Extract features based on user input
    "b" is to use baseline matching (intensity values of middle 9*9 pixels) as features
    "hm" is to use single normalized color histogram for whole image as features
    "chm" is to use single normalized color histogram for middle 200*200 pixels as features
    "ts" is to use whole image texture histogram based on sobel magnitude and orientation as features.
    "tl" is to use whole image texture histogram based on laws filters as features.
    "gf" is to use whole image texture histogram based on gabor filters as features.
    "mct" is to use combination of "hm" and "ts" for middle portion of image as features
    "sv" is to use single normalized histogram based on the spatial variance of color as features
    */

    if (strcmp(feature_type, "b") == 0) {
      features.clear();
      baseline_match(src,features);      
    }
    if (strcmp(feature_type, "hm") == 0){
      features.clear();
      int bins = 32; // no of bins for each axis of 2D histogram
      histogram_match(src,bins,features);
    } 
    if (strcmp(feature_type, "chm") == 0){
      features.clear();    
      int bins = 32; // no of bins for each axis of 2D histogram
      centre_image(src,bins,features);
    } 
    if (strcmp(feature_type, "ts") == 0){
      features.clear();
      int bins = 8; // no of bins for each axis of 2D histogram
      texture(src,bins,features);
    }
    if (strcmp(feature_type, "tl") == 0){
      features.clear();
      int bins = 8; // no of bins for  1D histogram 
      laws(src,bins,features);
    }
    if (strcmp(feature_type, "gf") == 0){
      features.clear();
      int bins = 8;  // no of bins for  1D histogram    
      gaborfilter(src,bins,features);
    }
    if (strcmp(feature_type, "mct") == 0){
      features.clear();
      int bins = 8; // no of bins for each axis of 2D histogram    
      midcolortext(src,bins,features);
    }
    if (strcmp(feature_type, "sv") == 0){
      features.clear();            
      spacialVariance(src,features);
      std::cout<<"check \n";
    }
    /*
    Given a filename, and image filename, and the image features, by
    default the function will append a line of data to the CSV format
    file.  If reset_file is true, then it will open the file in 'write'
    mode and clear the existing contents.

    The image filename is written to the first position in the row of
    data. The values in image_data are all written to the file as
    floats.

    The function returns a non-zero value in case of an error.
  */
    append_image_data_csv(filename, buffer, features, 0); 
    image_count++;

    }
  }  
  printf("Terminating\n");
  std::cout<<"Total no of images read"<<image_count<<"\n";
  return(0);
}


