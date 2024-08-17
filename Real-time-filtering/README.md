# Project Description

The project utilizes the OpenCV library in C++ to perform various tasks related to image and video processing. 

Run "imgDisplay.cpp" to read and display an image from a file.

Run "vidDisplay.cpp" for displaying live video from a webcam, and applying special effects.Special effects include
 * grayscale conversion
 * blurring and Sobel filtering
 * gradient magnitude calculation
 * color quantization
 * cartoonization
 * pixelation (extension) 
 * creating multiple sketch versions of the video (extension)

Users can choose which effects to apply by using different keystrokes and also have the ability to save an image with the applied special effects from the live video. 

The project mainly uses custom functions for applying the special effects, with the exception of the built-in functions for generating sketch versions of the video.

## Requirements


The project is tested in the following environment

* ubuntu 20.04

* VScode 1.74.3

* cmake 3.16.3

* g++ 9.4.0

* opencv 3.4.16


## Installation

For running "imgDisplay.cpp", edit line no 10 in CMakeLists.txt to
* add_executable( vidDisplay imgDisplay.cpp)


In the terminal

```bash
cd Project_1
mkdir build 
cd build
cmake ..
make
./vidDisplay $image_filename
```

For running "vidDisplay.cpp", edit line no 10 in CMakeLists.txt to
* add_executable( vidDisplay vidDisplay.cpp filter.cpp)

In the terminal

```bash
cd Project_1
mkdir build 
cd build
cmake ..
make
./vidDisplay 
```


## Usage


### Run the executables

For "imgDisplay.cpp"
* Type "s" to save an image  to your device in the working folder.
* Type "q" to close the  exit the program.

For "vidDisplay.cpp"
  * Type "g" to display default greyscale version of video.

  * Type "h" to display an alternative greyscale version of video."

  * Type "b" to display a blurred version (guassian) of video."

  * Type "x" to display edges on video in X direction (Sobel X)."

  * Type "y" to display edges on video in Y direction (Sobel Y)."

  * Type "m" to display edges on video(gradient magnitude using Sobel X and Sobel Y."

  * Type "l" to display a blur and quantized version of video."

  * Type "c" to display live cartoonization of video."  

  * Type "u" to increase the brightness of the image."

  * Type "d" to decrease the brightness of the image."

  * Type "r" to increase the contrast of the image."

  * Type "i" to decrease the contrast of the image." 

  * Type "q" to close the webcam and exit the program."

  * Type "z" to display default video channel from webcam."

  * Type "s" to save an image with current effect to your device."

  * Type "1" to apply edgeperserving filter to video."

  * Type "2" to enhance details (detail enhancement filter)on the video."

  * Type "3" to create a pencil sketch version of the video."

  * Type "4" to create a color pencil sketch version of the video."

  * Type "5" to create a water color sketch version of the video."

  * Type "p" to display pixelized version of video."

  * Type "s" to save an image with current effect to your device."

## Acknowledgements

Color conversions. https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray

Szeliski, R. (2010). Computer vision: Algorithms and applications. Springer. http://szeliski.org/Book/download/1TN5RZRLUADA/Szeliski_CVAABook_2ndEd.pdf

Mallick, S. (n.d.). Non-Photorealistic Rendering using OpenCV and Python. https://learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/
