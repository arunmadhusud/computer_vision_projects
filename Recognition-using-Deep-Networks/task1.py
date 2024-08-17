# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : task A to E
# This code loads the MNIST digit dataset, builds a convolutional network, trains the network and saves the model


# import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from network import Mynetwork
from network import train_network
from network import test_network

'''
Get the MNIST digit dataset
look at the first six example digits
Build a network model
Train the network
save the model
'''
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False
    # define the hyperparameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10 
    filter_size =5
    dropout_rate = 0.5   
    # get the MNIST dataset
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])), batch_size=batch_size_train, shuffle=True)
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

    # display the first six example digits    
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])        
    plt.show() 

    # build the network
    network = Mynetwork(filter_size, dropout_rate)
    # define the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum) 
    # initialize the loss vectors 
    train_losses = []
    train_counter = []
    test_losses = []    
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # train the network    
    test_network(network, test_loader, test_losses, test_counter)
    for epoch in range(1, n_epochs + 1):
        # train the network
        train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter)
        # test the network
        test_network(network, test_loader, test_losses, test_counter)
    
    # plot the loss
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # save the model
    torch.save(network.state_dict(), 'model_state_dict.pth')
    torch.save(network, 'model.pth')    
    return

if __name__ == "__main__":
    main(sys.argv)





