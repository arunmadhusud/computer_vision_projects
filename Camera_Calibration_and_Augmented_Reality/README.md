# Project Description

The objective of this project is to develop  a system that utilizes computer vision techniques and augmented reality technology to project virtual objects onto physical targets via live video stream. The system employs a calibrated camera that uses the corners of the target object as reference points for accurate tracking of the target's pose. The targets used in the system are chessboards and Aruco boards. The cameras were calibrated using corners extracted from multiple images of a chessboard. Additionally, a separate program has been developed to explore the use of feature detection algorithms such as Harris corners in augmented reality.

## Demo links

Demo of projecting virtual objects on chessboard:
https://northeastern-my.sharepoint.com/:v:/g/personal/madhusudhanan_a_northeastern_edu/EflOFViBgixFkfIbUyCAvTUB_N_z9WfwR4pDt3bURC-12Q?e=7JA4Nu

Demo of projecting virtual objects on Aruco board:
https://northeastern-my.sharepoint.com/:v:/g/personal/madhusudhanan_a_northeastern_edu/EfqoRfUQp1JPsqU1N0OqvQcBVwsSccHDhIIm4AqSArOIWg?e=0vktpb

Demo of detecting harris corners
https://northeastern-my.sharepoint.com/:v:/g/personal/madhusudhanan_a_northeastern_edu/EW6N5WTBtqFPhqD__cjVfQEBTn-I0QKMzmGGihsPaadCPw?e=KwRbAe


## Requirements


The project is tested in the following environment

* ubuntu 20.04

* VScode 1.74.3

* cmake 3.16.3

* g++ 9.4.0

* opencv 4.2


## Installation

For running "calibration.cpp", uncomment following line in CMakeLists.txt. 
* add_executable( calib calibration.cpp definitions.cpp)

In the terminal, type the commands

```bash
cd project_4
mkdir build 
cd build
cmake ..
make
./calib <directory path>
```
For running "arsystem.cpp", uncomment following line in CMakeLists.txt. 
* add_executable( calib arsystem.cpp definitions.cpp)

In the terminal, type the commands

```bash
cd project_4
mkdir build 
cd build
cmake ..
make
./calib <calibration_data_yaml_file> <imagefile>
```

For running "harriscorners.cpp", uncomment following line in CMakeLists.txt. 
* add_executable( calib harriscorners.cpp definitions.cpp)

In the terminal, type the commands

```bash
cd project_4
mkdir build 
cd build
cmake ..
make
./calib
```

directory path : folder location where the images used for calibration and yaml file with claibration data to be stored

calibration_data_yaml_file : location of yaml file in which calibration data (intrinsic parameters) are stored

imagefile : location of the image file to be superimposed on the target


## Usage


### To run the executables

For 'calibration.cpp'

1. Press 's' to save the images used for calibration
2. Press 'c' to calibrate the camera and store the data in a yaml file
3. Press 'q' to quit the program

For 'arsystem.cpp'

1. Press 't' to project a trapezoid on the chessboard 
2. Press 'c' to project a chair on the target 
3. Press 'l' to project a l-shape on the target 
4. Press 'f' to unproject any object or image from the target
5. Press 'q' to quit the program

For 'harriscorners.cpp'

1. Press 'h' to detect harris corners in the pattern 
2. Press 'r' to go back to default mode
2. Press 'q' to quit the program


### To run the extensions

1. Sytem can detect multiple targets at once. No additional input required for this from user.

2. System is used for calibration of multiple devices.

3. Aruco board is implemented as one of the target in 'arsystem.cpp'. Press 'l' to project a l-shape on the aruco board target.

4. An image can be overlayed on the chessboard target in 'arsystem.cpp'. Press 'o' to superimpose a picture on the chessboard  




## Acknowledgements

[1] Camera calibration using Aruco Boards https://docs.opencv.org/4.2.0/da/d13/tutorial_aruco_calibration.html

[2] Superimpose images over target
https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/

[3] Camera calibration using Chess Boards
https://docs.opencv.org/4.2.0/d4/d94/tutorial_camera_calibration.html

[4] Harris Corners using OpenCV
https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html



## No of Extension Days Used : 1