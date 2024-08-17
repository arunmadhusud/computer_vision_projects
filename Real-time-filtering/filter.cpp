//Arun_Madhusudhanan
//Project_1 spring 2023
//library for custom effects

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include "filter.h"

//function for generating alternative greyscale image
//compute average color values of all 3 channels of input image (src) and assign the average value to all 3 channels of output image(dst)
int greyscale( cv::Mat &src, cv::Mat &dst){
  //allocate dst image
  dst = cv::Mat::zeros( src.rows,src.cols, CV_8UC1 ); // signed short data type
  for(int i=0; i<=src.rows-1; i++){
    //src pointer
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);        
    for (int j=0;j<=src.cols-1;j++){
      uchar grey = (srcrptr[j][0])/3 + (srcrptr[j][1])/3+ (srcrptr[j][2])/3;          
      dst.at<uchar>(i,j) = grey;   
    
    }
  }
  return (0);
}

//function for implementing a 5x5 Gaussian filter as separable 1x5 filters
int blur5x5( cv::Mat &src, cv::Mat &dst ){
  //allocate a temporary image for storing horizontal filtered image
  cv::Mat tmp;
  tmp = cv::Mat::zeros( src.rows,src.cols,CV_8UC3);
  //allocate output image
  dst = cv::Mat::zeros( src.rows,src.cols,CV_8UC3);
  //applying_horizontal_filtering
  //[1,2,4,2,1] is sperable 1*5 filter used  
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstrptr = tmp.ptr<cv::Vec3b>(i);
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(j==0){
          dstrptr[j][c] = (1*srcrptr[j+1][c]+2*srcrptr[j][c]+4*srcrptr[j][c]+2*srcrptr[j+1][c]+1*srcrptr[j+2][c])/10; //padding by mirror
        }
        else if(j==1){
          dstrptr[j][c] = (1*srcrptr[j-1][c]+2*srcrptr[j-1][c]+4*srcrptr[j][c]+2*srcrptr[j+1][c]+1*srcrptr[j+2][c])/10; //padding by mirror
        }
        else if(j==src.cols-1){
          dstrptr[j][c] = (1*srcrptr[j-2][c]+2*srcrptr[j-1][c]+4*srcrptr[j][c]+2*srcrptr[j][c]+1*srcrptr[j-1][c])/10; //padding by mirror
        }
        else if(j==src.cols-2){
          dstrptr[j][c] = (1*srcrptr[j-2][c]+2*srcrptr[j-1][c]+4*srcrptr[j][c]+2*srcrptr[j+1][c]+1*srcrptr[j+1][c])/10; //padding by mirror
        }
        else dstrptr[j][c] = (1*srcrptr[j-2][c]+2*srcrptr[j-1][c]+4*srcrptr[j][c]+2*srcrptr[j+1][c]+1*srcrptr[j+2][c])/10;
      }
    }
  } 
  //applying_vertical_filtering
  //[1,2,4,2,1]' is sperable 1*5 filter used
  for(int i=0;i<=src.rows-1;i++){
    cv::Vec3b *srcrptr = tmp.ptr<cv::Vec3b>(i);
    cv::Vec3b *srcrptrm1 = tmp.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *srcrptrm2 = tmp.ptr<cv::Vec3b>(i-2);
    cv::Vec3b *srcrptrp1 = tmp.ptr<cv::Vec3b>(i+1);
    cv::Vec3b *srcrptrp2 = tmp.ptr<cv::Vec3b>(i+2);
    cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
    for (int j=0;j<=src.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        if(i==0){          
          dstrptr[j][c] = (1*srcrptrp1[j][c]+2*srcrptr[j][c]+4*srcrptr[j][c]+2*srcrptrp1[j][c]+1*srcrptrp2[j][c])/10; //padding by mirror         
        }
        else if(i==1){
          dstrptr[j][c] = (1*srcrptrm1[j][c]+2*srcrptrm1[j][c]+4*srcrptr[j][c]+2*srcrptrp1[j][c]+1*srcrptrp2[j][c])/10; //padding by mirror
        }
        else if(i==src.rows-1){
          dstrptr[j][c] = (1*srcrptrm2[j][c]+2*srcrptrm1[j][c]+4*srcrptr[j][c]+2*srcrptr[j][c]+1*srcrptrm1[j][c])/10; //padding by mirror
        }
        else if(i==src.rows-2){
          dstrptr[j][c] = (1*srcrptrm2[j][c]+2*srcrptrm1[j][c]+4*srcrptr[j][c]+2*srcrptrp1[j][c]+1*srcrptrp1[j][c])/10; //padding by mirror
        }
        else dstrptr[j][c] = (1*srcrptrm2[j][c]+2*srcrptrm1[j][c]+4*srcrptr[j][c]+2*srcrptrp1[j][c]+1*srcrptrp2[j][c])/10;
       
      }
    }
  } 

  return(0);
  
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

//function that blurs and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
 //allocate output image
 dst = cv::Mat::zeros( src.rows,src.cols,CV_8UC3);
 //blurring the input image
 blur5x5(src,dst);
 for(int i=0;i<=dst.rows-1;i++){    
    cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
    int bucket_size = 255 / levels; //calculating bucket size
    int xt;
    for (int j=0;j<=dst.cols-1;j++){
      // for each color channel
      for(int c=0;c<3;c++) {
        xt = dstrptr[j][c] / bucket_size;
        dstrptr[j][c] = xt * bucket_size;
      }
    }
 }
 return(0);
}

//live video cartoonization function using the gradient magnitude and blur/quantize filters.
int cartoon( cv::Mat &src, cv::Mat &dst, int levels, int magThreshold ){
 cv::Mat quantized; //allocating quanitzed image
 cv::Mat grad_x; //allocating x sobel image
 cv::Mat grad_y; //allocating y sobel image
 cv::Mat grad_xy; //allocating magnitude gardient image
 sobelX3x3( src, grad_x ); //perform x sobel on input image
 sobelY3x3( src, grad_y ); //perform y sobel on input image
 magnitude(grad_x ,grad_y,grad_xy); //perform gradient magnitude using inputs from x sobel and y sobel
 blurQuantize(src, quantized, levels); //perform blur and quantization on input image
 //allocate output image
 dst = cv::Mat::zeros( quantized.rows,quantized.cols,CV_8UC3);
 for(int i=0;i<=quantized.rows-1;i++){    
    cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
    cv::Vec3b *grad_xyrptr = grad_xy.ptr<cv::Vec3b>(i);
    cv::Vec3b *quantrptr = quantized.ptr<cv::Vec3b>(i);    
    for (int j=0;j<=quantized.cols-1;j++){
      // for each color channel
      //Pixel values with a magnitude threshold greater than a given threshold value is set to black.
      for(int c=0;c<3;c++) {  
        if(grad_xyrptr[j][c]>=magThreshold){
          dstrptr[j][c] = 0;
        }
        else dstrptr[j][c]=quantrptr[j][c];  
        
      }
    }
 }
 return(0);

}

//function to apply pixelation to the frame
int pixelate(cv::Mat& src, cv::Mat& dst, int pixel_size){
  //allocate dst image
  dst = cv::Mat::zeros( src.rows,src.cols,CV_8UC3);
  for(int y=0;y<=src.rows-1;y+=pixel_size){     
    for(int x=0;x<=src.cols-1;x+=pixel_size){      
      int avg_b = 0;
      int avg_g = 0;
      int avg_r = 0;
      int count=0;
      //takes the average of the color values of all pixels within a region defined by the pixel size 
      //and the current loop variables, and then applies that average color value to all pixels within that region in the "dst" image.     
      for(int i=y; i < y + pixel_size && i < src.rows; i++){
        cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);        
        for (int j = x; j < x + pixel_size && j < src.cols; j++) {
          avg_b += srcrptr[j][0];
          avg_g += srcrptr[j][1];
          avg_r += srcrptr[j][2];
          count++;
        }      
      }
      avg_b /= count;
      avg_g /= count;
      avg_r /= count;
      for(int i=y; i < y + pixel_size && i < src.rows; i++){
        cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
        for (int j = x; j < x + pixel_size && j < src.cols; j++) {
          dstrptr[j][0] = avg_b;
          dstrptr[j][1] = avg_g;
          dstrptr[j][2] = avg_r;
        }      
      }
    }

  }
return(0);
}

//function to adjust brightness and contrast of the frame
int brightness_contrast(cv::Mat &src, cv::Mat &dst,double alpha,int beta){
  dst = cv::Mat::zeros( src.rows,src.cols,src.type());
  for(int i =0; i<src.rows; i++){
    cv::Vec3b *srcrptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
    if (alpha<=0) alpha = 0.01;
    for (int j=0; j<src.cols; j++){
      for (int c=0; c<3;c++){
        if((alpha*srcrptr[j][c] + beta)>=255) dstrptr[j][c] = 255; //Any values going above 255 is set us white
        else if ((alpha*srcrptr[j][c] + beta)<=0) dstrptr[j][c] = 0; //values going below 0 is set us black
        else dstrptr[j][c] = alpha*srcrptr[j][c] + beta;
      }      
    }
  }
  return(0);
}


