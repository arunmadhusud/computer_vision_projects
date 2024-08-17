//Arun_Madhusudhanan
//Project_1 spring 2023
//Description provided below
// "Hello! This program lets you to apply different effects to the video from your webcam."
//  "Please select from following options."
//  "g) to display default greyscale version of video."
//  "h) to display an alternative greyscale version of video."
//  "b) to display a blurred version (guassian) of video."
//  "x) to display edges on video in X direction (Sobel X)."
//  "y) to display edges on video in Y direction (Sobel Y)."
//  "m) to display edges on video(gradient magnitude using Sobel X and Sobel Y."
//  "l) to display a blur and quantized version of video."
//  "c) to display live cartoonization of video."
//  "p) to display pixelized version of video."
//  "u) to increase the brightness of the image."
//  "d) to decrease the brightness of the image."
//  "r) to increase the contrast of the image."
//  "i) to decrease the contrast of the image."
//  "1) to apply edgeperserving filter to video."
//  "2) to enhance details (detail enhancement filter)on the video."
//  "3) to create a pencil sketch version of the video."
//  "4) to create a color pencil sketch version of the video."
//  "5) to create a water color sketch version of the video."
//  "z) to display default video channel from webcam."
//  "s) to save an image with current effect to your device."
//  "q) to close the webcam and exit the program."
//  "Have fun."


#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "filter.h"


int main(int argc, char *argv[]){

 //Open webcam video channel
 cv::VideoCapture *cap;
 cap = new cv::VideoCapture(0);
 // get some properties of the image
 cv::Size refS( (int) cap->get(cv::CAP_PROP_FRAME_WIDTH ),
 (int) cap->get(cv::CAP_PROP_FRAME_HEIGHT));

//declared flags for all the custom effects here
 bool is_greyscale=0;  //flag for greyscale image
 bool is_custom_greyscale=0; //flag for alternative greyscale image
 bool is_blur=0; //flag for guassian filter
 bool is_xsobel=0; //flag for x sobel
 bool is_ysobel=0; //flag for y sobel
 bool is_grad=0; //flag for gradient magnitude
 bool is_blurQuantize=0; //flag for blur and quantization
 bool is_cartoon=0; //flag for cartoonization
 bool is_pixel=0; //flag for pixelization
 bool is_increase_bright=0; //flag for increasing the brightness
 bool is_increase_contrast=0; //flag for increasing the contrast
 bool is_decrease_bright=0; //flag for decreasing the brightness
 bool is_decrease_contrast=0; //flag for decreasing the contrast
 bool is_edgepreservesmoothing=0; //flag for applying edge perserving filter
 bool is_detailenhancement=0; //flag for detail enhancement filter
 bool is_pencilsketch=0; //flag for pencil sketch filter
 bool is_colorpencilsketch=0; //flag for color pencil sketch filter
 bool is_stylization=0; //flag for water color sketch filter
 

 //declared variables here 
 int increase_bright_count=0;
 int decrease_bright_count=0;
 int increase_contrast_count=0;
 int decrease_contrast_count=0;
 int levels; //bucket size for quantization of image
 int magThreshold; //threshold value for cartoonization of image
 double alpha = 1; //initialize the multiplication factor (gain) to change contrast
 int beta=0; //initialize the addition factor (bias) to change brightness
 int pixel_size =25; //initialize the pixel bucket size for pixelazation of image

 //Display options to the user
 std::cout<<"Hello! This program lets you to apply different effects to the video from your webcam.\n";
 std::cout<<"Please select from following options.\n";
 std::cout<<"g) to display default greyscale version of video.\n";
 std::cout<<"h) to display an alternative greyscale version of video.\n";
 std::cout<<"b) to display a blurred version (guassian) of video.\n";
 std::cout<<"x) to display edges on video in X direction (Sobel X).\n";
 std::cout<<"y) to display edges on video in Y direction (Sobel Y).\n";
 std::cout<<"m) to display edges on video(gradient magnitude using Sobel X and Sobel Y. \n";
 std::cout<<"l) to display a blur and quantized version of video.\n";
 std::cout<<"c) to display live cartoonization of video.\n";
 std::cout<<"p) to display pixelized version of video.\n";
 std::cout<<"u) to increase the brightness of the image.\n";
 std::cout<<"d) to decrease the brightness of the image.\n";
 std::cout<<"r) to increase the contrast of the image.\n";
 std::cout<<"i) to decrease the contrast of the image.\n";
 std::cout<<"1) to apply edgeperserving filter to video.\n";
 std::cout<<"2) to enhance details (detail enhancement filter)on the video.\n";
 std::cout<<"3) to create a pencil sketch version of the video.\n";
 std::cout<<"4) to create a color pencil sketch version of the video.\n";
 std::cout<<"5) to create a water color sketch version of the video.\n";
 std::cout<<"z) to display default video channel from webcam,\n";
 std::cout<<"s) to save an image with current effect to your device.\n";
 std::cout<<"q) to close the webcam and exit the program.\n";
 std::cout<<"Have fun.\n";

 //verify if video channel is open
 if(!cap->isOpened()){
   std::cout<<"Not able oprn the camera device.\n";
   exit(-1);
 }


 
 //identifies a display window
 cv::namedWindow("Video_Display",1);
 

 //loop for capturing, manipulating and displaying frames
 while(true){
    cv::Mat src; // input_frame
    cv::Mat dst; // output_frame
    cv::Mat sx;  // gradient_magnitude image from Sobel X
    cv::Mat sy;  // gradient_magnitude image from Sobel Y

    //capture frame     
    *cap>> src;
    

    //check if frame is empty   
    if(src.empty()){
        std::cout<<"frame is empty.\n";
        break;
    }
    
    //display greyscale video based on the user input
    if (is_greyscale){
        //convert image to greyscale
        cv::cvtColor(src,dst,cv::COLOR_BGR2GRAY);
        cv::imshow("Video_Display",dst);
    }

    //display alternetive greyscale video based on the user input
    else if (is_custom_greyscale){
        //convert image to alternative greyscale 
        greyscale(src, dst);        
        cv::imshow("Video_Display",dst);
    }

    //display blurred(5x5 Gaussian filter) video based on the user input
    else if (is_blur){
        blur5x5(src,dst);
        cv::imshow("Video_Display",dst);
    }

    //display X sobel video based on the user input
    else if (is_xsobel){
        sobelX3x3( src, dst );
        cv::convertScaleAbs(dst,dst,2);
        cv::imshow("Video_Display",dst);
    }

    //display Y sobel video based on the user input
    else if (is_ysobel){
        sobelY3x3( src, dst );
        cv::imshow("Video_Display",dst);
        cv::convertScaleAbs(dst,dst,2);
        cv::imshow("Video_Display",dst);
    }

    //display gardient magnitude video (from X sobel and Y sobel) based on the user input
    else if (is_grad){
        sobelX3x3( src, dst );
        dst.copyTo(sx);
        sobelY3x3( src, dst );
        dst.copyTo(sy);
        magnitude(sx,sy,dst);        
        cv::imshow("Video_Display",dst);      
        
    }

    //display blur and quantized video based on the user input
    else if (is_blurQuantize){
        blurQuantize( src, dst, levels=15 );//size of bucket used =15
        cv::imshow("Video_Display",dst);
    }

    //display live video based on the user input
    else if (is_cartoon){        
        cartoon( src, dst, levels=15, magThreshold=18);
        cv::imshow("Video_Display",dst);
    }

    //display pixelised video based on the user input (extension)
    else if(is_pixel){        
        pixelate(src,dst, pixel_size);
        cv::imshow("Video_Display",dst);
    } 
        

    //display video with increased brightness based on the user input 
    else if(is_increase_bright){
        if(increase_bright_count==0){
            beta += 5;                     
            increase_bright_count++;
        }
        brightness_contrast(src,dst,1,beta);
        cv::imshow("Video_Display",dst);
    }

    //display video with decreased brightness based on the user input 
    else if(is_decrease_bright){
        if(decrease_bright_count==0){
            beta -= 5;                       
            decrease_bright_count++;
        }
        brightness_contrast(src,dst,1,beta); 
        cv::imshow("Video_Display",dst);
    }

    //display video with increased contrast based on the user input
    else if(is_increase_contrast){
        if(increase_contrast_count==0){
            alpha += 0.1;       
            increase_contrast_count++;
        }
        brightness_contrast(src,dst,alpha,0);
        cv::imshow("Video_Display",dst);
    }

    //display video with decreased contrast based on the user input
    else if(is_decrease_contrast){
        if(decrease_contrast_count==0){
            alpha -= 0.1;            
            decrease_contrast_count++;
        }
        brightness_contrast(src,dst,alpha,0);
        cv::imshow("Video_Display",dst);
    }

    //apply an edgeperserving filter and display based on the user input (extension)
    else if(is_edgepreservesmoothing){        
        cv::edgePreservingFilter(src,dst,1);
        cv::imshow("Video_Display",dst);
    }

    //apply a detail enhancing filter and display based on the user input (extension)
    else if(is_detailenhancement){        
        cv::detailEnhance(src,dst);
        cv::imshow("Video_Display",dst);
    }

    //produces a display that looks like a pencil sketch (extension)
    else if(is_pencilsketch){        
        cv::Mat dst_color;
        cv::pencilSketch(src,dst,dst_color,10 , 0.1f, 0.03f);
        cv::imshow("Video_Display",dst);
    }

    //produces a display that looks like a color pencil sketch (extension)
    else if(is_colorpencilsketch){
        cv::Mat dst_gray;        
        cv::pencilSketch(src,dst_gray,dst,10 , 0.1f, 0.03f);
        cv::imshow("Video_Display",dst);
    }

    //produces a display that looks like a water sketch (extension)
    else if(is_stylization){        
        cv::stylization(src,dst);
        cv::imshow("Video_Display",dst);
    }    

    //display unfiltered video if the user hasn't selected any option   
    else cv::imshow("Video_Display",src);

    //check for user input
    char key = cv::waitKey(10);
    
    //program quit if the user types 'q'
    if (key=='q') break;

    //save an image to a file if the user types 's'
    else if (key=='s'){        
        std::string filename = "./save.jpg";            // name of the output image file        
        if(dst.empty()) cv::imwrite(filename, src);
        else cv::imwrite(filename, dst);
        std::cout<<"Image saved as save.jpg in your current directory.\n";        
    }
    
    //set the flags based on the user input
    //if the user types 'g' it displays a greyscale version of the image instead of color. 
    else if(key=='g'){
        is_greyscale=1; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
              
    }

    //if the user types 'z' it displays the default video channel from webcam. 
    else if(key=='z'){
        is_greyscale=0; 
        is_custom_greyscale=0; 
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0; 
        is_blurQuantize=0;
        is_cartoon=0; 
        is_pixel=0;               
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0; 
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;            
           
    }

    //if the user types 'h' it displays an alternative greyscale version of the image (calculated using average color of all 3 channels).
    else if (key=='h'){
        is_custom_greyscale=1;
        is_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;                
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
    }

    // if the user types 'b' it displays a blurred version (guassian) of the image (in color). 
    else if (key=='b'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=1;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
    }

    //if the user types 'x' it displays X sobel of the image.
    else if (key=='x'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=1;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
    }

    //if the user types 'y' it displays Y sobel of the image.
    else if (key=='y'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=1;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
    }

    //if the user types 'm' it displays gradient magnitude ( from x sobel and y sobel) of the image.
    else if (key=='m'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=1;
        is_blurQuantize=0;
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        
    }

    //if the user types 'l' it displays blur and quantized version of the image.
    else if (key=='l'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=1;
        is_cartoon=0;        
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;        

    }

    //if the user types 'c' it displays cartoonized version of the image.
    else if (key=='c'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=1;        
        is_pixel=0;                
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0; 
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;         
    }

    //if the user types 'p' it displays pizelised version of the image(extension).
    else if(key=='p'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;        
        is_pixel=1;                
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0; 
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;              
        
    }
    
    //if the user types 'u' it increases the brightness of the image.
    else if(key=='u'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;        
        is_pixel=0;
        increase_bright_count=0;        
        is_increase_bright=1;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        increase_bright_count=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;              

    }

    //if the user types 'd' it decreases the brightness of the image.
    else if(key=='d'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;       
        is_pixel=0;     
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=1;
        is_decrease_contrast=0;        
        decrease_bright_count=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;             

    }

    //if the user types 'r' it increases the contrast of the image.
    else if(key=='r'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;        
        is_pixel=0;      
        is_increase_bright=0;
        is_increase_contrast=1;
        is_decrease_bright=0;
        is_decrease_contrast=0;        
        increase_contrast_count=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;               

    }

    //if the user types 'i' it decreases the contrast of the image.
    else if(key=='i'){
        is_greyscale=0;
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0;
        is_cartoon=0;        
        is_pixel=0;       
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=1;        
        decrease_contrast_count=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;       

    }

    //if the user types '1' it applies edgeperserving filter to the image.
    else if(key=='1'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=1;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;      
    }

    //if the user types '2' it applies detail enhancement filter to the image.
    else if(key=='2'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=1;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=0;       
    }

    //if the user types '3' it creates a pencil sketch version of  the image.
    else if(key=='3'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=1;
        is_colorpencilsketch=0;
        is_stylization=0;        
    } 

    //if the user types '4' it creates a color pencil sketch version of  the image.
    else if(key=='4'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=1;
        is_stylization=0;        
    }

    //if the user types '5' it creates a water color sketch sketch version of  the image.
    else if(key=='5'){
        is_greyscale=0; 
        is_custom_greyscale=0;
        is_blur=0;
        is_xsobel=0;
        is_ysobel=0;
        is_grad=0;
        is_blurQuantize=0; 
        is_cartoon=0;
        is_pixel=0;        
        is_increase_bright=0;
        is_increase_contrast=0;
        is_decrease_bright=0;
        is_decrease_contrast=0;
        is_edgepreservesmoothing=0;
        is_detailenhancement=0;
        is_pencilsketch=0;
        is_colorpencilsketch=0;
        is_stylization=1;        
    }   
    
    
 }

 // Release video channel and close window
 cv::destroyWindow("Video_Display");
 return(0);
}
