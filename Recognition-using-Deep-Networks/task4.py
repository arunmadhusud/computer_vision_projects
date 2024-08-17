## Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Task 4
# This code loads the MNIST dataset, trains the network  but with different parameters and saves the model
# Extension 1 : Experimented with 4 parameters

'''
Experiment with different parameters and report the results in a text file, the loss curve is saved as a png file
The parameters to experiment with are:
1. Number of epochs, values to try: 5,10,15
2. Batch size, values to try: 32,64,128
3. Dropout rate, values to try: 0.3 and 0.5
4. Filter size, values to try: 3 and 5
'''

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
This function trains the network with the given parameters
@param batch_size_train: batch size for training
@param batch_size_test: batch size for testing
@param learning_rate: learning rate for the optimizer
@param momentum: momentum for the optimizer
@param log_interval: number of batches after which the training loss is printed
@param n_epochs: number of epochs
@param filter_size: size of the filter
@param dropout_rate: dropout rate
'''
def plan(batch_size_train, batch_size_test, learning_rate, momentum, log_interval, n_epochs, filter_size, dropout_rate):      
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

    # build the network
    network = Mynetwork(filter_size, dropout_rate)
    # define the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)  
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    # file name to save the results after eaach experiment
    file_name = "task4.txt"
    
    # train and test the network
    test_network(network, test_loader, test_losses, test_counter,False)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter)
        test_loss, accuracy = test_network(network, test_loader, test_losses, test_counter,True)
    
    # save the results in a text file
    with open(file_name, "a") as f:
        f.write("n_epochs: "+str(n_epochs)+" batch_size_train: "+str(batch_size_train)+" dropout_rate: "+str(dropout_rate)+" filter_size: "+str(filter_size)+"\n")
        f.write('Epoch:{} \tTest set: Avg. loss: {:.4f}, Accuracy:({:.3f}%)\n'.format(epoch,test_loss, accuracy))
    
    # save the loss curve as a png file
    image_file_name = "task4_n_epochs_"+str(n_epochs)+"_batch_size_train_"+str(batch_size_train)+"_dropout_rate_"+str(dropout_rate)+"filter_size_"+str(filter_size)+".png"  
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(image_file_name)
    plt.close()   
    return test_loss, accuracy

# main function 
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False
    # define the parameters to experiment with
    n_epochs = [20]
    batch_size_train = [32,64,128]
    dropout_rate = [0.3,0.4,0.5]
    # dropout_rate = [0.4]
    filter_size = [3,5,7]
    # filter_size = [7]
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10 
    
    # initialize the best parameters
    test_loss_best = 5
    accuracy_best = 10

    # initialize the worst parameters
    test_loss_worst = 0
    accuracy_worst = 100


    # experiment with different parameters
    for n in n_epochs:
        for b in batch_size_train:
            for f in filter_size:
                for d in dropout_rate:                   
                    print("n_epochs: ",n,"batch_size_train: ",b,"dropout_rate: ",d,"filter_size: ",f,"\n")
                    test_loss,accuracy = plan(b, batch_size_test, learning_rate, momentum, log_interval, n, f, d) 
                    # save the best parameters
                    if test_loss < test_loss_best:
                        test_loss_best = test_loss
                        accuracy_fr_best_test_loss = accuracy
                        n_fr_best_test_loss= n
                        b_fr_best_test_loss = b
                        f_fr_best_test_loss= f
                        d_fr_best_test_loss= d  
                    if accuracy > accuracy_best:
                        accuracy_best = accuracy
                        test_loss_fr_best_accuracy = test_loss
                        n_fr_best_accuracy= n
                        b_fr_best_accuracy = b
                        f_fr_best_accuracy= f
                        d_fr_best_accuracy= d
                    
                    # save the worst parameters
                    if test_loss > test_loss_worst:
                        test_loss_worst = test_loss
                        accuracy_fr_worst_test_loss = accuracy
                        n_fr_worst_test_loss= n
                        b_fr_worst_test_loss = b
                        f_fr_worst_test_loss= f
                        d_fr_worst_test_loss= d
                    if accuracy < accuracy_worst:
                        accuracy_worst = accuracy
                        test_loss_fr_worst_accuracy = test_loss
                        n_fr_worst_accuracy= n
                        b_fr_worst_accuracy = b
                        f_fr_worst_accuracy= f
                        d_fr_worst_accuracy= d
    print("\n")
    # print the best parameters
    print("best test loss: ",test_loss_best,"accuracy: ",accuracy_fr_best_test_loss,"n: ",n_fr_best_test_loss,"b: ",b_fr_best_test_loss,"f: ",f_fr_best_test_loss,"d: ",d_fr_best_test_loss)
    print("best accuracy: ",accuracy_best,"test loss: ",test_loss_fr_best_accuracy,"n: ",n_fr_best_accuracy,"b: ",b_fr_best_accuracy,"f: ",f_fr_best_accuracy,"d: ",d_fr_best_accuracy)
    print("\n")
    # print the worst parameters
    print("worst test loss: ",test_loss_worst,"accuracy: ",accuracy_fr_worst_test_loss,"n: ",n_fr_worst_test_loss,"b: ",b_fr_worst_test_loss,"f: ",f_fr_worst_test_loss,"d: ",d_fr_worst_test_loss)
    print("worst accuracy: ",accuracy_worst,"test loss: ",test_loss_fr_worst_accuracy,"n: ",n_fr_worst_accuracy,"b: ",b_fr_worst_accuracy,"f: ",f_fr_worst_accuracy,"d: ",d_fr_worst_accuracy)
    return

if __name__ == "__main__":
    main(sys.argv)
    
        