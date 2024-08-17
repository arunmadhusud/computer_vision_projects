# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : task F to G
# This code reads the saved model and runs it on the first 10 test set images and on hand drawn images


#import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from network import Mynetwork
from torchvision import datasets, transforms

'''
Read the network and run it on the first 10 test set images and on hand drawn images
Plot first 9 digits and their predictions as a 3x3 grid
'''

'''
find the label of the image
@param loader: the test set loader
@param model: the trained model
@param num_images: the number of images to run the model on
'''
def find_label(loader,model,num_images):
    inp_images = []
    pred_labels = []
    i=0
    for data, target in loader:
        if i < num_images:
            # squeeze the data to get the image
            squeeze = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            # append the image to the list
            inp_images.append(squeeze)
            # run the model on the image
            with torch.no_grad():
                output = model(data)
                print(f'Output value of {i+1}th digit : {output}')
                print(f'Index of max output value of {i+1}th digit : {output.argmax().item()}')
                # get the label of the image
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'Correct label of {i+1}th digit : {label}')
                # append the label to the list
                pred_labels.append(label)
                i += 1
    # plot the first 9 images and their predictions as a 3x4 grid
    for i in range(num_images-1):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(inp_images[i][:,:,0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(pred_labels [i]))
        plt.xticks([])
        plt.yticks([])       
    plt.show()
    return 




def main(argv):
    # set the print options
    torch.set_printoptions(precision=2)
    # make the code repeatable
    torch.manual_seed(42)
    # Load the model
    model = torch.load('model.pth')
    # Test the model
    model.eval()
    
    # task F
    # load test data
    test_loader = DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=True)
    
    # initialize the lists for the first 9 images and their labels
    num_images = 10
    ten_images = []
    ten_labels = []
    i = 0
    # run the model on the first 10 test set images
    find_label(test_loader,model,num_images)
    
   

    # task G
    # Test the model on handwritten digits
    # load test data
    image_dir = "/home/arun/PRCV/project_5/handwritten"
    images_loader = DataLoader(datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])))
    
    num_images = 11
    # run the model on the first 10 test set images
    find_label(images_loader,model,num_images)      
    return

if __name__ == "__main__":
    main(sys.argv)


