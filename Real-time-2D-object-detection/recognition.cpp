/*
Arun_Madhusudhanan
Project_3 spring 2023
This file contains the functions used for 2D object recognition.
*/


#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "recognition.h"

/* This function is used to apply thresholding on the image
*  @param src: input image
*  @param dst: output image
*/
int thresholding(cv::Mat &src,cv::Mat &dst){
    dst = cv::Mat::zeros(src.size(),CV_8UC1);
    cv::Mat imageblur;
    cv::GaussianBlur(src,imageblur,cv::Size(3,3),cv::BORDER_REFLECT); //apply 3*3 guassian filter to blur the image
    // blur5x5(src,imageblur);
    cv::Mat imagehsv;
    cv::cvtColor(imageblur,imagehsv,cv::COLOR_BGR2HSV); //convert the image to HSV color space
    for(int i=0; i<=imagehsv.rows-1; i++){    
    cv::Vec3b *hsvrptr = imagehsv.ptr<cv::Vec3b>(i);
        for (int j=0;j<=imagehsv.cols-1;j++){
        // for each color channel
            for(int c=0;c<3;c++) {
                if (hsvrptr[j][1] > 70) {   //if saturation is greater than 50, then
                    hsvrptr[j][2] = hsvrptr[j][2] * 0.7; //reduce the intensity value by 60%. This is done to reduce the effect of illumination                
                }               
            }
        }
    }
    cv::Mat processed;
    cv::cvtColor(imagehsv,processed,cv::COLOR_HSV2BGR); //convert the image back to BGR color space  
    for(int i=0; i<=dst.rows-1; i++){        
        uchar *dstrptr = dst.ptr<uchar>(i); 
        cv::Vec3b *grayrptr = processed.ptr<cv::Vec3b>(i);
        for (int j=0;j<=dst.cols-1;j++){        
            if ((grayrptr[j][0]+ grayrptr[j][1]+grayrptr[j][2]) > 100 * 3) dstrptr[j]= 0; //thresholding, a pixel is considered as black if the sum of its RGB values is greater than 100*3
            else dstrptr[j]= 255;  //else it is considered as white                 
        }
    } 
return (0); 
}


/*This function is used to apply erosion on the image. 
* @param src: input image
* @param dst: output image
* @param erosion_elem: type of structuring element
* @param erosion_size: factor to control the size of structuring element. size = 2*erosion_size + 1
* @param erosion_iter: number of iterations
*/
//https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
//https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
int erosion(cv::Mat &src,cv::Mat &dst,int erosion_elem,int erosion_size, int erosion_iter){    
    int erosion_type = 0; 
    if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; } //MORPH_RECT: a rectangular structuring element (8 neighbors)
    else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; } //MORPH_CROSS: a cross-shaped structuring element (4 neighbors)
    else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; } //MORPH_ELLIPSE: an elliptic structuring element
    cv::Mat element = cv::getStructuringElement( erosion_type,cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),cv::Point( -1,-1) );
    cv::erode( src, dst, element,cv::Point( -1,-1),erosion_iter);
    return (0);
}


/* This function is used to apply dilation on the image. 
* @param src: input image
* @param dst: output image
* @param dilation_elem: type of structuring element
* @param dilation_size: factor to control size of structuring element. size = 2*dilation_size + 1
* @param dilation_iter: number of iterations
*/
int dilation(cv::Mat &src,cv::Mat &dst,int dilation_elem,int dilation_size,int dilation_iter){    
    int dilation_type = 0; 
    if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; } //MORPH_RECT: a rectangular structuring element (8 neighbors)
    else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; } //MORPH_CROSS: a cross-shaped structuring element (4 neighbors)
    else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; } //MORPH_ELLIPSE: an elliptic structuring element
    cv::Mat element = cv::getStructuringElement( dilation_type,cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),cv::Point( -1,-1) );
    cv::dilate( src, dst, element,cv::Point( -1,-1),dilation_iter);
    return (0);
}


/*This function is used to apply generate random color to apply for each region
* @param usedColors: vector of colors already used
* @return color: random color
*/
cv::Scalar generateRandomColor(std::vector<cv::Scalar>& usedColors) {
    cv::Scalar color;
    int count = 0;     
    while(true){
        color = cv::Scalar(100 + rand() % 156, 100 + rand() % 156, 100+ rand() % 156);
        count = std::count(usedColors.begin(), usedColors.end(), color);
        if (count == 0) {
         usedColors.push_back(color);         
         break;
        }
    }
    return color;    
}

/*This function is used to apply connected component analysis on the image. Uses a union find algorithm to find the connected components.(coded from scratch, extension 1)
* @param src: input image
* @param reg: output image
*/
int connectedcomponnect(cv::Mat &src,cv::Mat &reg){    
    cv::Mat dst;    
    dst = cv::Mat::zeros(src.size(),CV_16SC1);    
    int counter = 1;
    std::vector<int> regions;
    std::vector<cv::Scalar> usedColors;
    int size = src.rows * src.cols;
    int unionfind[size];    
    std::fill_n(unionfind, size, -1);     
    for(int i=1; i<src.rows-1; i++){        
        uchar *srcrptr = src.ptr<uchar>(i);
        uchar *srcrptrm1 = src.ptr<uchar>(i-1);        
        short *dstrptr = dst.ptr<short>(i);
        short *dstrptrm1 = dst.ptr<short>(i-1);                  
        for (int j=1;j<src.cols-1;j++){            
            if(srcrptr[j]==255) {                        
                if (srcrptrm1[j]==0 && srcrptr[j-1]==0) {
                    dstrptr[j]= counter;
                    counter++;                                   
                }
                else if (dstrptrm1[j]==0 && dstrptr[j-1]!=0) {
                    dstrptr[j]= dstrptr[j-1];                    
                }
                else if (dstrptrm1[j]!=0 && dstrptr[j-1]==0) {
                    dstrptr[j]= dstrptrm1[j];
                }
                else {
                    int min= std::min(dstrptrm1[j],dstrptr[j-1]);
                    int max= std::max(dstrptrm1[j],dstrptr[j-1]);
                    dstrptr[j]= min;
                    if(min != max){
                            unionfind[max] = min;
                    }
                }
            }            
        }        
    }   
    for(int i=1; i<src.rows-1; i++){ 
        short *dstrptr = dst.ptr<short>(i);
        for (int j=1;j<src.cols-1;j++){ 
            if (dstrptr[j] != 0 && unionfind[dstrptr[j]] != -1) {
                dstrptr[j] = unionfind[dstrptr[j]];
                while(unionfind[dstrptr[j]] != -1){
                    dstrptr[j] = unionfind[dstrptr[j]];                                     
                }
                // std::cout << "counter: " << int(dstrptr[j]) << std::endl;  
                
            }
        }
    }
    std::vector<int> regioncounter(counter,0);
    for(int i=1; i<src.rows-1; i++){ 
        short *dstrptr = dst.ptr<short>(i);
        for (int j=1;j<src.cols-1;j++){ 
            if (dstrptr[j] != 0) {
                regioncounter[dstrptr[j]]++;
            }
        }
    }   
    
    cv::Scalar color ;    
    for (int k=0;k<=regioncounter.size()-1;k++){
        if (k!=0 && regioncounter[k] > 500) {
            regions.push_back(k);
            // std::cout << "region " << k << " has " << regioncounter[k] << " pixels" << std::endl;
            color = generateRandomColor(usedColors);
            for(int i=1; i<src.rows-1; i++){
                cv::Vec3b *regrptr = reg.ptr<cv::Vec3b>(i);
                short *dstrptr = dst.ptr<short>(i); 
                for (int j=1;j<src.cols-1;j++){
                    if (dstrptr[j] == k) {                        
                        regrptr[j][0] = color[0];
                        regrptr[j][1] = color[1];
                        regrptr[j][2] = color[2];                        
                    }                    
                }
               
            }            
        }
    }

    for(int i=1; i<src.rows-1; i++){        
        uchar *srcrptr = src.ptr<uchar>(i);
        cv::Vec3b *regrptr = reg.ptr<cv::Vec3b>(i);         
        for (int j=1;j<src.cols-1;j++){            
            if(srcrptr[j]==0) { 
                regrptr[j][0] = 0;
                regrptr[j][1] = 0;
                regrptr[j][2] = 0;
            }
        }
    }
    return (0);
}


/* This function is used to find the connected components in the image using in built function
@param src: input image
@param reg: output image with regions identified
@param labelimage: image with region labels marked
@param valid_labels: vector of valid labels. Regions with pixels less than min_pixel_count are not considered as valid labels
@param centroids: centroids of the regions
*/
void connectedcomponent(cv::Mat &src,cv::Mat &reg,cv::Mat &labelimage,std::vector<int> &valid_labels,cv::Mat &centroids){
    cv::Mat stats;    
    int nLabels = cv::connectedComponentsWithStats(src, labelimage, stats, centroids, 8, CV_32S);    
    int min_pixel_count = 2000;
    std::vector<std::pair<int, int>> label_counts;
    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
        // Count the number of pixels in the current label
        int pixel_count = 0;
        for(int r = 0; r < labelimage.rows; ++r){
            for(int c = 0; c < labelimage.cols; ++c){
                if(labelimage.at<int>(r, c) == label){
                    pixel_count++;
                }
            }
        }
        // If the number of pixels is less than the minimum, set the color to background
        if(pixel_count < min_pixel_count){
            colors[label] = colors[0];        
        }
        else {
            // Store the label and its pixel count in the vector of pairs
            label_counts.push_back(std::make_pair(label, pixel_count));            
            // valid_labels.push_back(label);
            }
    }
    // Sort the vector of pairs in descending order of pixel count
    std::sort(label_counts.begin(), label_counts.end(), [](const std::pair<int, int> &left, const std::pair<int, int> &right){
        return left.second > right.second;
    });

   // Push the first 3 labels to the valid_labels vector
    int count = 0;
    for(auto const &label_count : label_counts){
        if(count >= 3 || label_count.second < min_pixel_count){
            break;
        }
        valid_labels.push_back(label_count.first);
        count++;
    }

    // Set the output image to the color of the corresponding label
    for(int r = 0; r < reg.rows; ++r){
        for(int c = 0; c < reg.cols; ++c){
            int label = labelimage.at<int>(r, c);
            if (std::find(valid_labels.begin(), valid_labels.end(), label) != valid_labels.end()) {
                cv::Vec3b &pixel = reg.at<cv::Vec3b>(r, c);
                pixel = colors[label];
            } else {
                cv::Vec3b &pixel = reg.at<cv::Vec3b>(r, c);
                pixel = colors[0]; // set to background color
            }
        }
    }
}

/*This function calculates the standard deviation of each dimension in the feature vectors stored in csv file
* @param database_features: feature vectors of the database images
* @param std_devs: standard deviations of each dimension in the feature vectors
* @return: 0 if successful, -1 if error
*/
int standard_deviation(const std::vector<std::vector<float>> &database_features, std::vector<float> &std_devs) {
    // Calculate the standard deviation of each dimension in the feature vectors
    for (int i = 0; i < database_features[0].size(); i++) {
        double mean = 0;
        for (int j = 0; j < database_features.size(); j++) {
            mean += database_features[j][i];
        }
        mean /= database_features.size();

        double sum_of_sq_diff = 0;
        for (int j = 0; j < database_features.size(); j++) {
            sum_of_sq_diff += ((database_features[j][i] - mean) * (database_features[j][i] - mean));
        }
        double variance = sum_of_sq_diff / database_features.size();
        std_devs[i] = sqrt(variance);
    }
return (0);
}


/*This function calculates the scaled Euclidean distance between the target feature vector and each feature vector in the database
* @param target_feature: feature vector of the target image
* @param database_features: feature vectors of the database images
* @param std_devs: standard deviations of each dimension in the feature vectors
* @dist: scaled Euclidean distance between the target feature vector and each feature vector in the database
*/
int scaled_euclidean(std::vector<float> &target_feature, std::vector<float> &database_feature,std::vector<float> &std_devs,float &dist) { 
    for (int j = 0; j < target_feature.size(); j++) {
        float diff = target_feature[j] - database_feature[j];        
        dist += (diff * diff) / (std_devs[j] * std_devs[j]);
    } 
    dist = sqrt(dist);
    return 0;
}


/* This function finds the nearest neighbour of the target feature vector in the database. Uses the scaled Euclidean distance as distance metric
* @param labels: labels of the database images extrcted from csv
* @param database_features: feature vectors of the database images
* @param target_feature: feature vector of the target image
*/
int nearneighbour(std::vector<char *> &labels,std::vector<std::vector<float>> &database_features,std::vector<float> &target_feature,char* label){
    // Check if the size of the target feature vector matches the size of the first feature vector in the database
    if (target_feature.size() != database_features[0].size()) {
        std::cout << "Error: target feature vector size does not match size of feature vectors in database" << std::endl;
        return -1;
    }
    //set a threshold for the distance to be considered as a match
    float threshold = 2;
    // Calculate the standard deviation of each dimension in the feature vectors
    std::vector<float> std_devs(database_features[0].size(), 0);
    standard_deviation(database_features, std_devs);
    // for(int i=0;i<std_devs.size();i++){
    //     std::cout<<std_devs[i]<<" ";
    // }
    float min_dist = FLT_MAX;

    for (int i = 0; i < database_features.size(); i++) {
        float dist = 0;
        scaled_euclidean(target_feature, database_features[i], std_devs,dist);
        // std::cout<<dist<<std::endl;
        if (dist < min_dist) {
            min_dist = dist;
            // std::cout<<min_dist<<std::endl;
            std::strcpy(label, labels[i]);
        }
    }
    if(min_dist > threshold){
        std::strcpy(label, "unknown");
    }
    return 0;

}

/* This function finds the k nearest neighbour of the target feature vector in the database. Uses the scaled Euclidean distance as distance metric.
* @param labels: labels of the database images extracted from csv
* @param database_features: feature vectors of the database images
* @param target_feature: feature vector of the target image
* @param label: label of the nearest neighbour
* @param k: number of nearest neighbours to consider
*/
int KN_nearest(std::vector<char *> &labels, std::vector<std::vector<float>> &database_features, std::vector<float> &target_feature, char* label, int k){
    // Calculate the standard deviation of each dimension in the feature vectors
    std::vector<float> std_devs(database_features[0].size(), 0);
    standard_deviation(database_features, std_devs);

    //https://www.geeksforgeeks.org/unordered_map-in-cpp-stl/
    std::unordered_map<std::string, std::vector<float>> knn_map;
    for (int i = 0; i < database_features.size(); i++) {
        float dist = 0;
        scaled_euclidean(target_feature, database_features[i], std_devs,dist);
        std::string object(labels[i]);
        if(knn_map.find(object) == knn_map.end()){
            std::vector<float> tmp;
            tmp.push_back(dist);
            std::pair<std::string , std::vector<float>> map_member (object, tmp);
            knn_map.insert(map_member);
        }else{
            knn_map[object].push_back(dist);
        }
    }

    //https://www.geeksforgeeks.org/unordered_map-in-cpp-stl/
    float min_dist = FLT_MAX;
    // float min_dist = -1;
    //set a threshold for the distance to be considered as a match
    float threshold = 2;
    for(auto x: knn_map){
        std::vector<float> dist_values = x.second;
        if(dist_values.size()<k){
            continue;
        }
        std::sort(dist_values.begin(), dist_values.end());
        float dist_val = 0.0;
        for(int i=0; i<k; i++){
            dist_val += dist_values[i];
            // std::cout<<dist_values[i]<<" ";
        }
        dist_val = dist_val/k;        
        if(dist_val< min_dist){
            min_dist = dist_val;
            // std::cout <<min_dist<<std::endl;
            if(min_dist > threshold) std::strcpy(label, "unknown");
            else std::strcpy(label, x.first.c_str());
        }
    }

    return 0;
}

/*This function gets the class name of the object from the label entered by user.
if the label is not present in the map, it asks the user to enter the name of the object to be added in the map
* @param c: label of the object
* @return: name of the object
*/
std::string classname(char c,std::map<char, std::string> &label_map) {    

    if (label_map.find(c) == label_map.end()) {
        std::cout << "Enter the name of the unknown object: ";
        std::string name;
        std::getline(std::cin, name);
        label_map[c] = name;
        std::cout << "Labelled this unknown object as :" << name << std::endl;
    }

    return label_map[c];
}




















