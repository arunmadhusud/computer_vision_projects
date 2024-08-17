//Arun_Madhusudhanan
//Project_2 spring 2023
//library of functions for feature exraction and distance calculation.

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "matching.h"

// Given an image (src), use the 9x9 square in the middle of the image as a feature vector for baseline matching
int baseline_match(cv::Mat &src, std::vector<float> &features){  
    int row_centre = src.rows/2;
    int col_centre = src.cols/2;    
    for(int i=row_centre-4; i<=row_centre+4; i++){    
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);        
    for (int j=col_centre-4;j<=col_centre+4;j++){ 
    //add the RGB values of 9x9 square to feature vector           
    features.push_back(srcrptr[j][0]);
    features.push_back(srcrptr[j][1]);
    features.push_back(srcrptr[j][2]);
    }
  }
  return (0);
}

//function to sort filenames based on the increasing order of distance
int pairsort(std::vector<float> &distance, std::vector<char *> &filenames)
{
    int n = distance.size();    
    // Create a vector of pairs from distance and filenames vectors
    std::vector<std::pair<float, char *>> vec;
    for (int i = 0; i < n; i++) { 
        vec.push_back(std::make_pair(distance[i], filenames[i]));
    }    
    // Sort the vector of pairs based on the increasing order of distance
    std::sort(vec.begin(), vec.end());    
    // Update filenames vector based on the sorted vector of pairs
    for (int i = 0; i < n; i++) { 
        filenames[i] = vec[i].second; 
    }     
  return (0);
}

/*Given a RGB image, this function gives a single normalized rg chromaticity histogram as the
feature vector. No of bins for the histogram has to be passed as an argument.
*/
int histogram_match(cv::Mat &src,int &bins, std::vector<float> &features){
cv::Mat hist;
hist = cv::Mat::zeros(bins,bins,CV_32FC1); //initialize a 2D histogram with zeros
int r_bucket;
int g_bucket;
float r; //r chromaticity
float g; //g chromaticity
float hist_sum = 0; //sum to normalize histogram

//Create 2D histogram by looping through the image
for(int i=0;i<=src.rows-1;i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);    
    for (int j=0;j<=src.cols-1;j++){
        //find r and g values
        //if pixel values are zero, assigned a value of 0.33 to both r and g
        if(srcrptr[j][0] == 0 && srcrptr[j][1] == 0 && srcrptr[j][2] == 0){ 
            r = 0.33;
            g = 0.33;
        }
        else {
        r = (float)srcrptr[j][2]/(srcrptr[j][0]+srcrptr[j][1]+srcrptr[j][2]);
        g = (float)srcrptr[j][1]/(srcrptr[j][0]+srcrptr[j][1]+srcrptr[j][2]);
        }
        r_bucket = r * bins; //find the bin number for r value
        g_bucket = g * bins; //find the bin number for g value
        if (r_bucket == bins)  r_bucket = bins - 1;
        if (g_bucket == bins)  g_bucket = bins - 1;        
        float *histrptr = hist.ptr<float>(g_bucket);
        histrptr[r_bucket] = histrptr[r_bucket] + 1; //update the pixel count of the bin
        hist_sum +=1;  //update the value of sum by 1 everytime a pixel is added to a bin 
    }
    
  }

/*Normalization
loop through histogram to divde each bin value by total sum*/
for(int i=0;i<=hist.rows-1;i++){  
  float *histrptr = hist.ptr<float>(i);
  for (int j=0;j<=hist.cols-1;j++){
    histrptr[j] = (float)histrptr[j]/(hist_sum);    
  }
}

//add the values of 2D histogram to feature vector
for(int i=0;i<=hist.rows-1;i++){  
  float *histrptr = hist.ptr<float>(i);
  for (int j=0;j<=hist.cols-i-1;j++){
    features.push_back(histrptr[j]);    
  }
}
return (0);
}

/*Given a RGB image, this function gives a single normalized rg chromaticity histogram of centre
100 x 100 pixels as the feature vector. No of bins for the histogram has to be passed as an argument.
*/
int centre_image(cv::Mat &src,int &bins, std::vector<float> &features){  
  cv::Mat hist;  
  int row_centre = src.rows / 2;
  int col_centre = src.cols / 2;  
  hist = cv::Mat::zeros(bins,bins,CV_32FC1);  //initialize a 2D histogram with zeros
  int r_bucket;
  int g_bucket;
  float r; //r chromaticity
  float g; //g chromaticity
  float hist_sum = 0;  //sum to normalize histogram

  //Create 2D histogram by looping through the centre 100 x 100 pixels of image
  for(int i=row_centre - 100;i<=row_centre + 100;i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);      
    for (int j=col_centre - 100;j<=col_centre + 100;j++){
      //find r and g values
      //if pixel values are zero, assigned a value of 0.33 to both r and g      
      if(srcrptr[j][0] == 0 && srcrptr[j][1] == 0 && srcrptr[j][2] == 0){
          r = 0.33;
          g = 0.33;
      }
      else {
      r = (float)srcrptr[j][2]/(srcrptr[j][0]+srcrptr[j][1]+srcrptr[j][2]);
      g = (float)srcrptr[j][1]/(srcrptr[j][0]+srcrptr[j][1]+srcrptr[j][2]);
      }
      
      r_bucket = r * bins;  //find the bin number for r value
      g_bucket = g * bins;  //find the bin number for g value
      if (r_bucket == bins)  r_bucket = bins - 1;
      if (g_bucket == bins)  g_bucket = bins - 1;        
      float *histrptr = hist.ptr<float>(g_bucket); 
      histrptr[r_bucket] = histrptr[r_bucket] + 1;  //update the pixel count of the bin      
      hist_sum +=1; //update the value of sum by 1 everytime a pixel is added to a bin  
    }    
  }

  /*Normalization
  loop through histogram to divde each bin value by total sum*/  
  for(int i=0;i<=hist.rows-1;i++){    
    float *histrptr = hist.ptr<float>(i);
    for (int j=0;j<=hist.cols-1;j++){
      histrptr[j] = (float)histrptr[j]/(hist_sum);      
    }
  } 
  
  //add the values of 2D histogram to feature vector
  for(int i=0;i<=hist.rows-1;i++){   
    float *histrptr = hist.ptr<float>(i);
    for (int j=0;j<=hist.cols-i-1;j++){
      features.push_back(histrptr[j]);      
    }  
  }
  
  return (0);
}

//funtion for implementing 3x3 Sobel X  as separable 1x3 filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
  //allocate a temporary image for storing horizontal filtered image
  cv::Mat tmp;
  tmp = cv::Mat::zeros( src.rows,src.cols,CV_16SC3);
  //allocate output image
  dst = cv::Mat::zeros( src.rows,src.cols,CV_16SC3);
  //applying_horizontal_filtering
  //[-1,0,1] is sperable 1*3 filter used  
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *dstrptr = tmp.ptr<cv::Vec3s>(i);    
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(j==0){
          dstrptr[j][c] = (-1*srcrptr[j][c]+0*srcrptr[j][c]+1*srcrptr[j+1][c]); //padding by mirror
        }        
        else if(j==src.cols-1){
          dstrptr[j][c] = (-1*srcrptr[j-1][c]+0*srcrptr[j][c]+1*srcrptr[j][c]); //padding by mirror
        }
        else dstrptr[j][c] = (-1*srcrptr[j-1][c]+0*srcrptr[j][c]+1*srcrptr[j+1][c]);
      }
    }
  } 
  //applying_vertical_weightage values
  //weightage values used= [1 2 1]' 
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3s *srcrptr = tmp.ptr<cv::Vec3s>(i);
    cv::Vec3s *srcrptrm1 = tmp.ptr<cv::Vec3s>(i-1);
    cv::Vec3s *srcrptrp1 = tmp.ptr<cv::Vec3s>(i+1);    
    cv::Vec3s *dstrptr = dst.ptr<cv::Vec3s>(i);
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(i==0){          
          dstrptr[j][c] = (1*srcrptr[j][c]+2*srcrptr[j][c]+1*srcrptrp1[j][c])/4;   //padding by mirror       
        }
        else if(i==src.rows-1){
          dstrptr[j][c] = (1*srcrptrm1[j][c]+2*srcrptr[j][c]+1*srcrptr[j][c])/4; //padding by mirror
        }       
        else dstrptr[j][c] = (1*srcrptrm1[j][c]+2*srcrptr[j][c]+1*srcrptrp1[j][c])/4;        
      }
    }
  }

  return(0);
}

//funtion for implementing 3x3 Sobel Y  as separable 1x3 filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
  //allocate a temporary image for storing horizontal filtered image
  cv::Mat tmp;
  tmp = cv::Mat::zeros( src.rows,src.cols,CV_16SC3);
  dst = cv::Mat::zeros( src.rows,src.cols,CV_16SC3);
  //applying_horizontal_weightage values
  //weightage values used= [1 2 1]  
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *dstrptr = tmp.ptr<cv::Vec3s>(i);
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(j==0){
          dstrptr[j][c] = (1*srcrptr[j][c]+2*srcrptr[j][c]+1*srcrptr[j+1][c]); //padding by mirror
        }        
        else if(j==src.cols-1){
          dstrptr[j][c] = (1*srcrptr[j-1][c]+2*srcrptr[j][c]+1*srcrptr[j][c]); //padding by mirror
        }
        else dstrptr[j][c] = (1*srcrptr[j-1][c]+2*srcrptr[j][c]+1*srcrptr[j+1][c]);
      }
    }
  } 
  //applying_vertical_filtering
  //[1,0,-1] is sperable 1*3 filter used
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3s *srcrptr = tmp.ptr<cv::Vec3s>(i);
    cv::Vec3s *srcrptrm1 = tmp.ptr<cv::Vec3s>(i-1);
    cv::Vec3s *srcrptrp1 = tmp.ptr<cv::Vec3s>(i+1);    
    cv::Vec3s *dstrptr = dst.ptr<cv::Vec3s>(i);
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(i==0){          
          dstrptr[j][c] = (1*srcrptr[j][c]+0*srcrptr[j][c]+-1*srcrptrp1[j][c])/4;  //padding by mirror        
        }
        else if(i==src.rows-1){
          dstrptr[j][c] = (1*srcrptrm1[j][c]+0*srcrptr[j][c]+-1*srcrptr[j][c])/4; //padding by mirror
        }       
        else dstrptr[j][c] = (1*srcrptrm1[j][c]+0*srcrptr[j][c]+-1*srcrptrp1[j][c])/4;        
      }
    }
  }

  return(0);
}

//function that generates a gradient magnitude image from the X and Y Sobel images.
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
  //allocate output image
  dst = cv::Mat::zeros( sx.rows,sx.cols,CV_16SC3);
  for(int i=0;i<=sx.rows-1;i++){
    cv::Vec3s *sxrptr = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *syrptr = sy.ptr<cv::Vec3s>(i);
    cv::Vec3s *dstrptr = dst.ptr<cv::Vec3s>(i);
    for (int j=0;j<=sx.cols-1;j++){
      for(int c=0;c<3;c++){
        dstrptr[j][c] = sqrt((sxrptr[j][c]*sxrptr[j][c])+(syrptr[j][c]*syrptr[j][c]));
      }
    }
  }  
  cv::convertScaleAbs(dst,dst,2);   //converting to 8UC3 type
  return(0);
}

//function that generates a gradient direction image from the X and Y Sobel images.
int gradient_orientation( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
  //allocate output image
  dst = cv::Mat::zeros( sx.rows,sx.cols,CV_16SC3);
  for(int i=0;i<=sx.rows-1;i++){
    cv::Vec3s *sxrptr = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *syrptr = sy.ptr<cv::Vec3s>(i);
    cv::Vec3s *dstrptr = dst.ptr<cv::Vec3s>(i);
    for (int j=0;j<=sx.cols-1;j++){
      for(int c=0;c<3;c++){
        dstrptr[j][c] = atan2(syrptr[j][c],sxrptr[j][c]);
      }
    }
  }  
  cv::convertScaleAbs(dst,dst,2);   //converting to 8UC3 type
  return(0);
}

/* This function gnerates a whole image texture histogram as the feature vector
Given a RGB image, this function gives a single normalized 2D histogram of gradient orientation and magnitude as the
feature vector.  No of bins for the histogram has to be passed as an argument.
*/
int texture(cv::Mat &src, int &bins, std::vector<float> &features){
  cv::Mat hist;  
  cv::Mat grayscale;  
  cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY); //convert image to grayscale prior to texture analysis  
  cv::Mat mag; //allocates sobel magnitude image
  cv::Mat orien; //allocates sobel gradient orientation image
  cv::Mat sx;  // gradient_magnitude image from Sobel X
  cv::Mat sy;  // gradient_magnitude image from Sobel Y
  float hist_sum = 0; //sum to normalize histogram
  sobelX3x3( grayscale, mag ); //compute sobel x image
  mag.copyTo(sx);
  sobelY3x3( grayscale, mag ); //compute sobel y image
  mag.copyTo(sy);
  magnitude(sx,sy,mag); //compute gradient magnitude image
  gradient_orientation(sx,sy,orien);//compute gradient orientation image
  hist = cv::Mat::zeros(bins,bins,CV_32FC1); //initialize a 2D histogram with zeros
  int m_buck;
  int o_buck;

  /*Create 2D whole image texture histogram by looping through the texture image
   gradient magnitude and orientation is used to build 2D histogram
  */
  for(int i=0;i<=mag.rows-1;i++){
    uchar *magrptr = mag.ptr<uchar>(i); 
    uchar *orienrptr = orien.ptr<uchar>(i);
    for (int j=0;j<=mag.cols-1;j++){
      m_buck = magrptr[j] * (float)bins/255; //find the bin number for gradient magnitude
      o_buck = orienrptr[j] * (float) bins / (2* CV_PI); ////find the bin number for gradient orientation
      if (m_buck == bins)  m_buck = bins - 1;
      if (o_buck == bins)  o_buck = bins - 1;
      float *histrptr = hist.ptr<float>(m_buck);
      histrptr[o_buck] = histrptr[o_buck] + 1; //update the  count of the bin
      hist_sum +=1; //update the value of sum by 1 everytime a pixel is added to a bin 
    }
  }
  /*Normalization
  loop through histogram to divde each bin value by total sum*/ 
  for(int i=0;i<=hist.rows-1;i++){    
    float *histrptr = hist.ptr<float>(i);
    for (int j=0;j<=hist.cols-1;j++){
      histrptr[j] = (float)histrptr[j]/(hist_sum);      
    }
  }
  //add the values of 2D histogram to feature vector
  for(int i=0;i<=hist.rows-1;i++){  
    float *histrptr = hist.ptr<float>(i);
    for (int j=0;j<=hist.cols-i-1;j++){
      features.push_back(histrptr[j]); 

    }
  }
return (0);
}

//compute the 1*5 Law's filters
int laws_filter(cv::Mat &src, std::vector<int> values){
  src = cv::Mat::zeros(1,5,CV_32F);
  for(int j=0;j<=src.cols-1;j++){
    src.at<float>(0, j) = values[j];
           
  }
  return(0);
}

/*This function combine rotated Law's filters.
For instance, if there are results from applying the L5E5' and E5L5' filters,
this function compute the magnitude of both (square root of sum of squares).
*/
int filter_comb( cv::Mat &sx, cv::Mat &sy,cv::Mat &dst ) {  
  dst = cv::Mat::zeros( sx.rows,sx.cols,CV_16SC1); //allocate output image
  for(int i=0;i<=sx.rows-1;i++){
    short *sxrptr = sx.ptr<short>(i);
    short *syrptr = sy.ptr<short>(i);
    short *dstrptr = dst.ptr<short>(i);        
    for (int j=0;j<=sx.cols-1;j++){      
        dstrptr[j]= sqrt((sxrptr[j]*sxrptr[j])+(syrptr[j]*syrptr[j])); //compute magnitude       
    }
  }  
  cv::convertScaleAbs(dst,dst,2);   //converting to 8UC1 type
  return(0);
}

/* Given a laws filtered image,this function computes 1D histogram and store it as feature.
   No of bins for histogram has to be passed as an argument.
*/
int laws_hist(cv::Mat &src, int &bins,std::vector<float> &features){
  cv::Mat hist;
  hist = cv::Mat::zeros(1,bins,CV_32FC1); //initialize a 1D histogram with zeros 
  int y;   
  for(int i=0;i<=src.rows-1;i++){
    float *histrptr = hist.ptr<float>(0);
    uchar *srcrptr = src.ptr<uchar>(i);       
    for (int j=0;j<=src.cols-1;j++){   
      y = srcrptr[j] * (float)bins/255;  //find the bin number for intensity value          
      if (y==bins) y = bins - 1;        
      histrptr[y] = histrptr[y] + 1;   //update the  count of the bin    
    }
  }
  //add the values of 1D histogram to feature vector
  for(int i=0;i<=hist.rows-1;i++){    
    float *histrptr = hist.ptr<float>(0);
    for (int j=0;j<=hist.cols-1;j++){      
      features.push_back(histrptr[j]);          
    }
  }  
  return(0);
}

/*Given an RGB image, this function applies a combination of law's filters 
on image and extract the 1D histogram responses as texture feature. Combinations 
of (l5,e5),(e5,s5),(s5,w5),(r5,w5) law's filters are used. Pixel values of each 
filtered image is used to create a 1D histogram.
*/
int laws(cv::Mat &src,int &bins,std::vector<float> &features){
  std::vector<int> l5_values = {1, 4, 6, 4, 1}; //values for Level filter
  std::vector<int> e5_values = {-1, -2, 0, 2, 1}; //values for Edge filter
  std::vector<int> s5_values = {-1, 0, 2, 0, -1}; //values for Spot filter
  std::vector<int> w5_values = {-1, 2, 0, -2, 1}; //values for Wave filter
  std::vector<int> r5_values = {1, -4, 6, -4, 1}; //values for Ripple filter
  cv::Mat image;
  cv::cvtColor(src,image,cv::COLOR_BGR2GRAY); //convert image to greyscale image prior to texture analysis
  cv::Mat hist; 

  cv::Mat l5;
  laws_filter(l5,l5_values); //compute the 1*5 Level filter
  cv::Mat e5;
  laws_filter(e5,e5_values); //compute the 1*5 Edge filter
  cv::Mat s5;
  laws_filter(s5,s5_values); //compute the 1*5 Spot filter
  cv::Mat w5;
  laws_filter(w5,w5_values); //compute the 1*5 Wave filter
  cv::Mat r5;
  laws_filter(r5,r5_values); //compute the 1*5 Ripple filter
  
  //compute law filter kernels
  cv::Mat l5_l5 = (l5.t() * l5)/256; //compute l5l5 filter and normalize
  cv::Mat l5_e5 = (l5.t() * e5)/48;  //compute l5e5 filter and normalize
  cv::Mat e5_l5 = (e5.t() * l5)/48;  //compute e5l5 filter and normalize
  cv::Mat e5_s5 = (e5.t() * s5)/12;  //compute e5s5 filter and normalize
  cv::Mat s5_e5 = (s5.t() * e5)/12;  //compute s5e5 filter and normalize
  cv::Mat s5_w5 = (s5.t() * w5)/12;  //compute s5w5 filter and normalize
  cv::Mat w5_s5 = (w5.t() * s5)/12;  //compute w5s5 filter and normalize
  cv::Mat w5_r5 = (w5.t() * r5)/48;  //compute w5r5 filter and normalize
  cv::Mat r5_w5 = (r5.t() * w5)/48;  //compute r5w5 filter and normalize

  cv::Point anchor = cv::Point( -1, -1 ); //inputs for filter2D command
  double delta = 0; //inputs for filter2D command
  
  //apply law filter kernels
  cv::Mat filterl5_e5;
  cv::filter2D(image, filterl5_e5,-1, l5_e5,anchor, delta, cv::BORDER_REFLECT);    
  cv::Mat filtere5_l5;
  cv::filter2D(image, filtere5_l5,-1, e5_l5,anchor, delta, cv::BORDER_REFLECT);
  cv::Mat filtere5_s5;    
  cv::filter2D(image, filtere5_s5,-1, e5_s5,anchor, delta, cv::BORDER_REFLECT);
  cv::Mat filters5_e5;    
  cv::filter2D(image, filters5_e5,-1, s5_e5,anchor, delta, cv::BORDER_REFLECT);    
  cv::Mat filters5_w5;
  cv::filter2D(image, filters5_w5,-1, s5_w5,anchor, delta, cv::BORDER_REFLECT);
  cv::Mat filterw5_s5;
  cv::filter2D(image, filterw5_s5,-1, w5_s5,anchor, delta, cv::BORDER_REFLECT);
  cv::Mat filterw5_r5;
  cv::filter2D(image, filterw5_r5,-1, w5_r5,anchor, delta, cv::BORDER_REFLECT);
  cv::Mat filterr5_w5;
  cv::filter2D(image, filterr5_w5,-1, r5_w5,anchor, delta, cv::BORDER_REFLECT);
  
  /*combine laws filtered images.
  For instance, if there are results from applying the L5E5' and E5L5' filters,
  compute the magnitude of both (square root of sum of squares)
  */
  cv::Mat l5_e5_combined;
  filter_comb(filterl5_e5,filtere5_l5,l5_e5_combined);
  cv::Mat e5_s5_combined;
  filter_comb(filtere5_s5,filters5_e5,e5_s5_combined);
  cv::Mat s5_w5_combined;
  filter_comb(filters5_w5,filterw5_s5,s5_w5_combined);
  cv::Mat r5_w5_combined;
  filter_comb(filterr5_w5,filterw5_r5,r5_w5_combined);
  
  //compute 1D histogram of laws filtered images and add values to features
  laws_hist(l5_e5_combined, bins,features);    
  laws_hist(e5_s5_combined, bins,features);    
  laws_hist(s5_w5_combined, bins,features);    
  laws_hist(r5_w5_combined, bins,features);

  /*Normalization
  loop through feature to divde each value by total sum*/
  float sum = std::accumulate(features.begin(),features.end(),0);
  for(int i=0;i<=features.size()-1;i++){    
      features[i] = features[i] / (float)sum;
  } 

return (0);
}

/*Given an RGB image, this function applies a 48 gabor filters 
on image and extract the 1D histogram responses as texture feature.
48 gabor filters were obtained by changing sigma values ( no = 3) and 
orientations (no = 16). Pixel values of each filtered image is used to create a 
1D histogram.
*/
int gaborfilter(cv::Mat &src,int &bins,std::vector<float> &features) {  
  cv::Mat image;
  cv::cvtColor(src,image,cv::COLOR_BGR2GRAY); // convert image to grayscale
  
  float sigma[] = {2.0, 3.0, 5.0}; //chose 3 sigma values
  for (auto i : sigma) {
    // chnage the orientation values (total 16)
    for (int a = 0; a < 16; a++) {
      float thetta = a * CV_PI / 8;
      cv::Mat gaborKernel = cv::getGaborKernel( cv::Size(30,30), i, thetta, 10.0, 0.5, 0, CV_32F);
      cv::Mat gaborImage;        
      cv::filter2D(image, gaborImage, CV_32F, gaborKernel); // apply gabor filter      
      laws_hist(gaborImage, bins,features);  // generate 1D histogram for each result and add it as a texture vector    
    }
  }
  
  /*Normalization
    loop through feature to divde each bin value by total sum*/
  float sum = std::accumulate(features.begin(),features.end(),0);
    for(int i=0;i<=features.size()-1;i++){    
       features[i] = features[i] / (float)sum;
    }  
  return(0);    
}

/*Task 5 - Approach 1
Given an RGB image, this function computes 2D color histogram of the
middle part of image and add that as feature. Then it computes 2D texture
histogram based on sobel magnitude and orientation and add it to the feature. */
int midcolortext(cv::Mat &src,int &bins,std::vector<float> &features){
  //divide whole image into 3X3 grids.
  int x = src.cols / 3; 
  int y = src.rows / 3;

  cv::Mat middle = src(cv::Rect(x, y, x, y)).clone(); //extract middle part of image   
  histogram_match(middle,bins,features); //2D color histogram of the middle part of image
  texture(middle,bins, features); //2D texture histogram based on sobel magnitude and orientation.

  /*Normalization
  loop through feature to divde each bin value by total sum*/
  float sum = 0;
    for(int i=0;i<=features.size()-1;i++){ 
      if(std::isnan(features[i])) continue;   
      else sum = sum + features[i];
    }
    for(int i=0;i<=features.size()-1;i++){    
        features[i] = features[i]/(float)sum;
    }
  return (0);
}


int spacialVariance(cv::Mat &src, std::vector<float> &features){
    int size = 8*8*8;
    std::vector<float> hist= std::vector<float>(size, 0.0); //histogram of feature size. pixel counts will be addded here
    std::vector<float> x_val= std::vector<float>(size, 0.0); //total row number ( x value) of pixels for each bin will be stored here
    std::vector<float> y_val= std::vector<float>(size, 0.0); //total col number ( y value) of pixels for each bin will be stored here
    std::vector<float> dist = std::vector<float>(size, 0.0); //store the distance of a pixel location from its mean location
    std::vector<float> avg_x = std::vector<float>(size, 0.0);  //store the average x value of each bin here.
    std::vector<float> avg_y = std::vector<float>(size, 0.0); //store the average y value of each bin here.
    
    int range = 32; //255/8
    for(int i=0;i<src.rows; i++){
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            //find the bn values of each pixel
            int B_bin = rptr[j][0]/range;  
            int G_bin = rptr[j][1]/range;
            int R_bin = rptr[j][2]/range;
            int bin = B_bin*8*8 + G_bin*8 + R_bin;
            hist[bin] += 1; //update histogram bin count
            x_val[bin] += i; //add the x values
            y_val[bin] += j; //add the y values
        }
    } 
        
    for(int i=0; i<size; i++){
       // find the average x value of each bin 
        if (hist[i]==0) avg_x[i]=0;
        else avg_x[i] = x_val[i]/hist[i];
        // find the average y value of each bin 
        if (hist[i]==0) avg_y[i];
        else avg_y[i] = y_val[i]/hist[i];
    }
    
    //compute the distance of each pixel location from its mean location
    for(int i=0;i<src.rows; i++){
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            int B_bin = rptr[j][0]/range;  
            int G_bin = rptr[j][1]/range;
            int R_bin = rptr[j][2]/range;
            int bin = B_bin*8*8 + G_bin*8 + R_bin;
            dist[bin] += ((i-avg_x[bin])*(i-avg_x[bin]) + (j-avg_y[bin])*(j-avg_y[bin])); //computing variance
        }
    }

    features = std::vector<float>(size, 0.0);
    //compute the standard devaiation for each bins and add it to the feature vector
    for(int i=0; i<size; i++){
        features[i] = std::sqrt(dist[i]/hist[i]);//get the standard deviation for each bins
    }

    /*Normalization
    loop through feature to divde each bin value by total sum*/
    float sum = 0;
    for(int i=0;i<=features.size()-1;i++){ 
      if(std::isnan(features[i])) continue;   
      else sum = sum + features[i];
    }    
    for(int i=0;i<=features.size()-1;i++){    
        features[i] = features[i]/(float)sum;
    }
    
    return 0;
}



/*Distance metric for spatial variance.
used histogram intersection method. Cells with nan values were omitted from distance calculation*/
int specialVariance_distance(std::vector<float> &features,std::vector<float> &distance, std::vector<std::vector<float>> data){
  double hm_inter;
  for (int i=0; i<data.size(); i++) {
    hm_inter= 0;       
    for (int j=0; j<data[i].size(); j++) {        
        if(std::isnan(features[j]) || features[j]==0 || std::isnan(data[i][j]) || data[i][j]==0){
            continue;
        }
        else {
          hm_inter += MIN(features[j], data[i][j]) ;          
        }       
    }
  hm_inter = 1 - hm_inter;
  distance.push_back(hm_inter);
  }  
return (0);
}


//calculate sum-of-squared-difference of features for the distance metric
int sumofsqdiff(std::vector<float> &feature,std::vector<float> &distance, std::vector<std::vector<float>> data){
  for (int i=0; i<data.size(); i++) {
    double sumofsqdiff = 0;
    if (feature.size() != data[i].size()) {
      std::cout << "Error: target image features and database image features have different sizes" << std::endl;      
      return -1;
    }
    for (int j=0; j<data[i].size(); j++) {
        sumofsqdiff = sumofsqdiff + (feature[j] - data[i][j]) * (feature[j] - data[i][j]) ;         
    } 
distance.push_back(sumofsqdiff); 
  }
return (0);
}

//calculate histogram instersection of features for the distance metric
int hm_inter(std::vector<float> &feature,std::vector<float> &distance, std::vector<std::vector<float>> data){
 double hm_inter;
 for (int i=0; i<data.size(); i++) {
    hm_inter= 0;
    if (feature.size() != data[i].size()) {
      std::cout << "Error: target image features and database image features have different sizes" << std::endl;      
      return -1;
    }    
    for (int j=0; j<data[i].size(); j++) {
        hm_inter = hm_inter + MIN(feature[j],data[i][j]);        
    }
 hm_inter = 1 - hm_inter;
 distance.push_back(hm_inter);
 } 
 return (0);
}
