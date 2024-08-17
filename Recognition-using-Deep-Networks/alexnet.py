# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Extension 2
# This code loads the pretrained Alexnet model and evaluates the first two layers of the model

# import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from task2 import analyze_filters
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np

'''
This function loads the pretrained Alexnet model and evaluates the first two layers of the model
'''
def main(argv):
    # reference : https://pytorch.org/hub/pytorch_vision_alexnet/
    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    
    # Load the pretrained Alexnet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()
    # Get the first two layers of the Alexnet model
    conv1 = model.features[0]
    conv2 = model.features[3]
    # Define a model that uses the first two layers of the Alexnet model
    new_model = nn.Sequential(*list(model.features.children())[:2])

    # Preprocess the image
    input_image = Image.open(filename)
    preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input_image = np.transpose(torch.squeeze(input_tensor,1).numpy(),(1,2,0))
    
    # Evaluate the first two layers of the Alexnet model
    # Note that the same image is used for  analyzing the filters of the first two layers of the Alexnet model
    analyze_filters(conv1,input_image)        
    analyze_filters(conv2,input_image)
    
    return

if __name__ == "__main__":
    main(sys.argv)

    


    
