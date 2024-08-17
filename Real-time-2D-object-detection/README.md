# Project Description

The objective of the project is to detect 2D objects placed on a white surface in a translation, scale, and rotation invariant manner. The system created for this purpose has two modes: training and classification. In the training mode, the system will learn from a collection of images of objects with white backgrounds taken by a camera positioned directly above. The system will capture translational, scale, and rotational invariant characteristics from these trained images and store them in a csv file. In the classification mode, the system will use nearest neighbor or k nearest neighbor methods, based on user input, to compare the characteristics of the target object with the features in the database to find the closest match. If the closest match is not found, the system will switch back to training mode to add the unknown object to the user base, based on the user's input.

## Demo links

Training (sample video)
https://drive.google.com/file/d/1spLJT8_ohKMc-Lj54QVOOqUwuiArh2EE/view?usp=share_link

Nearest Neighbour classification (full video):
https://drive.google.com/file/d/1NGfmw2nyGvinlnC6ZS4FDD1Y4LxdSwwq/view?usp=sharing

K-NN classification (sample) :
https://drive.google.com/file/d/1JlzFAC6rj2VSe3HrLpcWrVTj5UpQaBO9/view?usp=sharing

System identifies an object which is not in the database as unknown and goes into
training mode. 
https://drive.google.com/file/d/1NGfmw2nyGvinlnC6ZS4FDD1Y4LxdSwwq/view?usp=sh
aring


## Requirements


The project is tested in the following environment

* ubuntu 20.04

* VScode 1.74.3

* cmake 3.16.3

* g++ 9.4.0

* opencv 3.4.16


## Installation


In the terminal, type the commands

```bash
cd project_3
mkdir build 
cd build
cmake ..
make
./objectrec <directory path> <csv filename> classifier_type operation_mode
```


## Usage


### To run the executables

directory path : folder location where the images are stored for training/classification. Images for training and classification should be stored in seperate folders.

csv filename : csv file in which database features to be stored for training or be read for classification

Training mode: use "t" as operation_mode. You can keep the classifier_type as "n" or "k"

Classification mode: Use "c" as operation_mode. Use "n" as classifier_type  for nearest neighbor approach. Use "k" as classifier_type  for K - nearest neighbor approach. Note that K = 3 is used in the system.

 System identifies an object which is not in the database as unknown and goes into training mode. No additional input required for this from user.



## Acknowledgements

[1] Professor Bruce Maxwell : author of csv_util files.

[2] Inbuilt Moment functions in open CV. //https://learnopencv.com/shape-matching-using-hu-moments-c-python/

[3] Map data structure. https://www.geeksforgeeks.org/map-associative-containers-the-c-standard-template-library-stl/

[4] Putting a text in image on OpenCv.
https://www.geeksforgeeks.org/write-on-an-image-using-opencv-in-cpp/



