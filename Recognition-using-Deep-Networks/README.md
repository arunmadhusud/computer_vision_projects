# Project Description

In this project, I built and trained a network to recognize the MNIST digit. The network is tested on both test set and custom inputs. The first layers of the network were analyzed to understand how the layers  process the data . Later, the network was modified and trained to detect Greek Letters.  The final task was to experiment by changing the different aspects of the network and how it affects the performance. 
A live digit recognition system based on the network is also implemented. Additionally, the first layer of the network was changed to Gabor filter bank to understand  how it would affect the performance. Another pre-trained network called alexnet is loaded and analyzes the first couple of layers to understand how it processes the data.


## Requirements


The project is tested in the following environment

* ubuntu 20.04

* VScode 1.74.3

* python 3.6


## To run the executables

For task 1 A-E(build and train network), enter the command in terminal

```bash
python3 task1.py
```

For task 1 F-G(run network on test set), enter the command in terminal

```bash
python3 task1F.py
```
For task 2 (Examine network), enter the command in terminal

```bash
python3 task2.py
```
For task 3 (Transfer learning on Greek letters), enter the command in terminal

```bash
python3 task3.py
```
For task 4 ( Design experiment), enter the command in terminal

```bash
python3 task4.py
```
## To run the Extensions



For extension 1, no additional input required. Four dimensions are changed in task 4.

For extension 2 (loading pre-trained AlexNet and evaluate its first couple of convolutional layers ),
```bash
python3 alexnet.py
```


For extension 3 (replace the first layer of the MNIST network with a Gabor filter bank) ,
```bash
python3 gabor_filter.py
```


For extension 4 (live video digit recognition application using the trained network.),
```bash
python3 live_recognition.py
```
## Acknowledgements

[1] MNIST digit recognition network implementation : https://www.hackersrealm.net/post/mnist-handwritten-digits-recognition-using-python

[2] AlexNet implementation: https://pytorch.org/hub/pytorch_vision_alexnet/

[3] Resizing images: https://picresize.com/


## No of Extension Days Used : 2