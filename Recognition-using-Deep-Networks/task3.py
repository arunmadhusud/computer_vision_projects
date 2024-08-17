# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Task 3
# This code loads the greek letter dataset, replaces the last layer of the MNIST digit recognition network with a new layer, trains the network and saves the model
# The network is trained to recognize three different greek letters: alpha, beta, and gamma

# import statements
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import cv2
from network import Mynetwork
from network import train_network
from network import test_network
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms

'''
Transform the RGB greek letter images to grayscale
and resize them to 28x28 pixels
and invert the colors
'''
class GreekTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28,28))
        x= torchvision.transforms.functional.invert(x)        
        return x

def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False
    # define the hyperparameters
    n_epochs = 18
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 2
    
    # define the greek letters
    greek_letters = ['alpha', 'beta', 'gamma']   
    # Load the model
    model = torch.load('model.pth')
    # freeze the network weights
    for param in model.parameters():
        param.requires_grad = False
    # replace the last layer with a new layer with 3 outputs
    model.fc2 = nn.Linear(50, 3)
    # print the model
    print(model)

    # Dataloader for the Greek Dataset
    greek = '/home/arun/PRCV/project_5/greek_train/greek_train'
    greek_train = DataLoader(torchvision.datasets.ImageFolder(greek, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), GreekTransform(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=5, shuffle=True)

    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)  

    # define the loss vector
    train_losses = []
    train_counter = []   
    
    # Train the network
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, model, greek_train, optimizer, log_interval, train_losses, train_counter)
        
    
    # Plot the loss
    plt.plot(train_counter, train_losses, color='blue')    
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # save the model
    torch.save(model.state_dict(), 'greek_model_state_dict.pth')
    torch.save(model, 'greek_model.pth') 

    # Test the network
    model.eval()
    # load test data
    image_dir = "/home/arun/PRCV/project_5/greek_test"
    images_loader = DataLoader(datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])))
    
    # load the test data which is cropped to 128 x 128 pixels
    # image_dir = "/home/arun/PRCV/project_5/greek_test_2"
    # images_loader = DataLoader(torchvision.datasets.ImageFolder(image_dir, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), GreekTransform(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])))
    
    inp_images = []
    pred_labels = []
    for data, target in images_loader:
        #squeeze the data        
        squeeze = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
        # append the image to the list
        inp_images.append(squeeze)
        with torch.no_grad():
            output = model(data)
            print("\n")
            print(f'Output value : {output}')
            print(f'Index of max output value: {output.argmax().item()}')
            Index = output.data.max(1, keepdim=True)[1].item()
            label = greek_letters[Index]
            print(f'Correct label : {label}')
            pred_labels.append(label)
    
    # plot the first 14 images and their predictions as a 6x3 grid
    for i in range(14):
        plt.subplot(6,3,i+1)
        plt.tight_layout()
        plt.imshow(inp_images[i][:,:,0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(pred_labels [i]))
        plt.xticks([])
        plt.yticks([])       
    plt.show()
    return
   
if __name__ == "__main__":
    main(sys.argv)