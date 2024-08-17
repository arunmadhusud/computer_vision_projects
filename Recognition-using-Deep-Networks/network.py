# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : CNN for MNIST digit recognition
# This code contains the function for building and training a convolutional network to solve the MNIST digit recognition task.

# import statements
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
define the network
the network has following layers:
first convolutional layer has 10 filters of size, filter_size x filter_size
Max pooling layer with kernel size 2 and ReLU activation
second convolutional layer has 20 filters of size, filter_size x filter_size
Dopout layer with dropout rate, dropout_rate
Max pooling layer with kernel size 2 and ReLU activation
Fully connected layer with 50 neurons and ReLU activation
Fully connected layer with 10 neurons and ReLU activation
'''
class Mynetwork(nn.Module):
    # define the layers
    def __init__(self,filter_size,dropout_rate):
        super(Mynetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=filter_size)
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        # calculate the input size of the first fully connected layer
        self.fc1_input_size = self.calculate_fc1_input_size()                  
        self.fc1 = nn.Linear(self.fc1_input_size , 50)
        self.fc2 = nn.Linear(50, 10)
    
    # define the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, self.fc1_input_size )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)        
        # apply log softmax to the output
        return F.log_softmax(x,1) 
       
    # calculate the input size of the first fully connected layer
    def calculate_fc1_input_size(self):
        # create a dummy input tensor
        x = torch.randn(1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # calculate the number of elements in the output tensor
        return x.numel()//x.size(0)

'''
This function trains the network for one epoch
@param epoch: the current epoch number
@param network: the network to be trained
@param train_loader: the dataloader for the training data
@param optimizer: the optimizer to be used for training
@param log_interval: the interval at which the training loss is to be printed and logged
@param train_losses: the list to which the training loss is to be appended
@param train_counter: the list to which the number of training samples seen is to be appended
'''
def train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter):
  # set the network to training mode
  network.train()
  # initialize the number of correct predictions to 0
  correct = 0
  # iterate over the training data
  for batch_idx, (data, target) in enumerate(train_loader):
    # forward pass
    output = network(data)
    # clear the gradients
    optimizer.zero_grad() 
    # calculate the loss   
    loss = F.nll_loss(output, target)
    # calculate the number of correct predictions
    correct += (output.argmax(1) == target).type(torch.float).sum().item()    
    loss.backward()
    optimizer.step()
    # log the training loss
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*len(data)) + ((epoch-1)*len(train_loader.dataset)))    
  #calculate the accuracy
  accuracy = 100. * correct / len(train_loader.dataset)
  print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset), accuracy))

'''
This function tests the network
@param network: the network to be tested
@param test_loader: the dataloader for the test data
@param test_losses: the list to which the test loss is to be appended
@param test_counter: the list to which the number of test samples seen is to be appended
@param save: if True, the test loss and accuracy are returned
'''
def test_network(network, test_loader, test_losses, test_counter,save = False):
  # set the network to evaluation mode
  network.eval()
  # initialize the test loss to 0
  test_loss = 0
  # initialize the number of correct predictions to 0
  correct = 0
  # iterate over the test data
  with torch.no_grad():
    for data, target in test_loader:
      # forward pass
      output = network(data)
      # calculate the loss
      test_loss += F.nll_loss(output, target, size_average=False).item()
      # calculate the number of correct predictions
      pred = output.data.max(1, keepdim=True)[1]      
      correct += pred.eq(target.data.view_as(pred)).sum()
  # calculate the average test loss
  test_loss /= len(test_loader.dataset)
  # log the test loss
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  if save:    
    return test_loss, 100. * correct / len(test_loader.dataset)