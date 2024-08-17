# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Extension 3
# This code loads the MNIST dataset, trains the network  but the first layer of the MNIST network is replaced with a Gabor filter


# import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from network import train_network
from network import test_network
import numpy as np
import cv2


'''
This function defines the network with the Gabor filter
The first layer of the network is replaced with a Gabor filter
'''
class Gabornetwork(nn.Module):
    def __init__(self,gabor):
        super(Gabornetwork, self).__init__()
        # conv1 is only used for calculating the input size of the first fully connected layer  
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # replace the first layer with a Gabor filter
        self.gabor = gabor        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)
        # calculate the input size of the first fully connected layer
        self.fc1_input_size = self.calculate_fc1_input_size()                  
        self.fc1 = nn.Linear(self.fc1_input_size , 50)
        self.fc2 = nn.Linear(50, 10)

        
    
    # define the forward pass
    def forward(self, x):
        # apply the Gabor filter
        images =[]
        for i in range(len(x)):
            kernals = []
            for j in self.gabor:
                image = cv2.filter2D(x[i][0].detach().numpy(), -1, j)
                H = np.floor(np.array(j.shape)/2).astype(np.int64)                
                image = image[H[0]:-H[0],H[1]:-H[1]]
                kernals.append(image)               
            images.append(kernals)
        
        x = torch.from_numpy(np.array(images))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, self.fc1_input_size )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,1) 

    # calculate the input size of the first fully connected layer
    def calculate_fc1_input_size(self):
        # create a dummy input tensor
        x = torch.randn(1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # calculate the number of elements in the output tensor
        return x.numel()//x.size(0) 

# function to generate 10 Gabor filters
def generate_gabor_filters():
    gabor_filters=[]
    # loop through 10 different orientations
    for orientatiion in np.arange(0, np.pi, np.pi / 10):        
        # create a gabor filter
        kernal = cv2.getGaborKernel((5, 5),1.0, orientatiion,np.pi/2.0, 0.5, 0, ktype=cv2.CV_32F)
        # normalize the filter
        kernal /= 1.5*kernal.sum()
        # add the filter to the list
        gabor_filters.append(kernal)
    return gabor_filters

def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False
    # define the hyperparameters for the network
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10 

    # get the MNIST dataset
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])), batch_size= batch_size_train, shuffle=True)
    test_loader = DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),batch_size=batch_size_test, shuffle=True)
    
    # look at the first six example digits
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    # create a gabor filter
    gabor_filters = generate_gabor_filters()
    
    # create a network with the gabor filter
    network = Gabornetwork(gabor_filters)  
    

    # define the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    # initialize the lists for plotting the loss  
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    # train and test the network
    test_network(network, test_loader, test_losses, test_counter)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter)
        test_network(network, test_loader, test_losses, test_counter)
    
    # plot the loss
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # save the model
    torch.save(network.state_dict(), 'gabor_model_state_dict.pth')
    torch.save(network, 'gabor_model.pth') 

    return

if __name__ == "__main__":
    main(sys.argv)