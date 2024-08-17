# Project Description

The goal of this project is to enhance understanding of how to manipulate and examine images on a pixel-by-pixel basis for image matching. The objective is to identify images within a database that have similar content to a target image using the general characteristics of the images such as color, texture, and their arrangement in space. This project provides an opportunity to work with various color spaces, histograms, spatial attributes, and texture characteristics.

Use "readfiles.cpp" to extract the features of images in database and to save the features in a csv file.

Use "feature.cpp" to extract the features of traget image, opens the features of images in database stored in csv file and compute the distance between features. It display top N matches

## Requirements


The project is tested in the following environment

* ubuntu 20.04

* VScode 1.74.3

* cmake 3.16.3

* g++ 9.4.0

* opencv 3.4.16


## Installation

For running "readfiles.cpp", edit line no 10 in CMakeLists.txt to
* add_executable( feature_extraction readfiles.cpp csv_util.cpp matching.cpp)


In the terminal

```bash
cd project_2
mkdir build 
cd build
cmake ..
make
./feature_extraction <featuretype> <directory_path to image database> <csv_filename>
```

For running "feature.cpp", edit line no 10 in CMakeLists.txt to
* add_executable( feature_extraction feature.cpp csv_util.cpp matching.cpp)

In the terminal

```bash
cd project_2
mkdir build 
cd build
cmake ..
make
./feature_extraction  <target_image_filename> <feature_type> <matching_method> <num_matching_images(N)> <database_feature_filename1> <database_feature_filename2> 
```


## Usage


### Run the executables

For "readfile.cpp", following are the options for feature type.
* "b" is to use baseline matching (intensity values of middle 9*9 pixels) as features
* "hm" is to use single normalized color histogram for whole image as features
* "chm" is to use single normalized color histogram for middle 200*200 * pixels as features
* "ts" is to use whole image texture histogram based on sobel magnitude and orientation as features.
* "mct" is to use combination of "hm" and "ts" for middle portion of image as features.
* "tl" is to use whole image texture histogram based on laws filters as features.
* "gf" is to use whole image texture histogram based on gabor filters as features.
* "sv" is to use single normalized histogram based on the spatial variance of color as features.


For "feature.cpp", following are the options for feature type.
  * "b" : Use baseline matching (intensity values of middle 9*9 pixels)    as features.Use sum of squared difference ("sq") as distance metric for better results
  * "hm" is to use single normalized color histogram for whole image as features.Use histogram instersection ("hi") as distance metric for better results.
  * "mhm": Use mutihistogram matching. Feature 1: single normalized color histogram for middle 200*200 pixels.Feature 2: use single normalized color histogram for whole image as features. Use histogram instersection ("hi") as distance metric for better results.
  * "ts": Use texture (based on sobel magnitude and orientation) and color histogram for matching.Use histogram instersection ("hi") as distance metric  for better results.
  * "mct":Use texture (based on sobel magnitude and orientation) and color histogram of centre portion for matching. Use histogram instersection ("hi") as distance metric for better results.

For "feature.cpp", following are the options for distance metric type. 
 * "sq" for sum of squared difference.
 * "hi" for histogram instersection. 
 * "tl": Use texture (based on laws filters) and color histogram for matching. Use histogram instersection ("hi") as distance metric for better results.
 * "gf": Use texture (based on gabor filters) and color histogram for matching. Use histogram instersection ("hi") as distance metric for better results.
 * "sv": Use spatial variance of color and  color histogram. Use histogram instersection ("hi") as distance metric  for better results


## Acknowledgements

[1] Professor Bruce Maxwell, author of csv_utill.cpp and skeleton readfiles.cpp used for storing and reading features to/from csv.

[2] Gabor filter, https://en.wikipedia.org/wiki/Gabor_filter

[3] Laws filter, https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect12.pdf


