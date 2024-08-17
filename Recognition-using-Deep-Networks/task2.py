# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Task 2
# This code contains the function for analyzing the filters of the first convolutional layer of the network.


# import statements
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import cv2
from network import Mynetwork

'''
This function analyzes the filters of the first convolutional layer of the network
@param conv: the first convolutional layer of the network
@param first_image: the first image in the test set
'''
def analyze_filters(conv, first_image):
    # show first 10 filters of convolutional layer
    # list of filters
    filters = []
    with torch.no_grad():
        for i in range(10):
            # Create a subplot for the filter
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            # Get the ith filter 
            fil_curr = conv.weight[i,0]
            # Add the filter to the list of filters
            filters.append(fil_curr)
            # Print the filter and shape
            print (f'Filter {i+1} has shape {fil_curr.shape}')
            # print current filter weights
            print (fil_curr)
            plt.imshow(fil_curr, interpolation='none')
            plt.title(f'Filter {i+1}')
            # Turn off the axis ticks to get a cleaner plot
            plt.xticks([])
            plt.yticks([])
        plt.show()    

    # apply the 10 filters to the first image in the test set
    with torch.no_grad():
        images = []
        for i in range(10):
            # get the ith filter
            images.append(filters[i])
            # apply the filter to the image
            fileterd_image= cv2.filter2D(np.array(first_image), ddepth=-1, kernel=np.array(filters[i]))
            images.append(fileterd_image)
        for i in range(20):
            plt.subplot(5, 4, i+1)
            plt.tight_layout()
            plt.imshow(images[i],cmap='gray', interpolation='none')
            if i%2 == 0:
                plt.title(f'Filter {i//2+1}')
            else:
                plt.title(f'Image {i//2+1}')
            # plt.title(f'Image {i+1}')
            # Turn off the axis ticks to get a cleaner plot
            plt.xticks([])
            plt.yticks([])
        plt.show()


# main function
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # Load the model
    model = torch.load('model.pth')
    # print the model
    print(model)    
    # Get the first convolutional layer
    conv = model.conv1   
    # load training data
    train_loader = DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])))
    
    # get the first image
    batch = next(iter(train_loader))
    first_image = batch[0][0]
    first_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    # analyze the filters
    analyze_filters(conv, first_image)
    return
    

if __name__ == "__main__":
    main(sys.argv)




