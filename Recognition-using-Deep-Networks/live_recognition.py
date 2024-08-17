# Arun Madhusudhanan
# Spring 2023
# CS5330_Recognition using Deep Networks : Extension 4
# This code loads the trained model and uses the webcam to recognize the digits live

# import statements
import cv2
from task1 import Mynetwork
import ssl
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def main(argv):
    # Load the model    
    trained_model = torch.load('model.pth')
    # Test the model
    trained_model.eval()
 
    # Load the webcam
    cap = cv2.VideoCapture(2)

    while True:
        # Capture frame-by-frame
        ret, read_frame = cap.read()
        
        if ret:
            # resize the frame to 28x28 pixels
            resized_frame = cv2.resize(read_frame, (28,28))            
            # change the frame to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)            
            # add a dimension to the frame
            gray_frame = np.expand_dims(gray_frame, 0)            
            # change the data type
            gray_frame = gray_frame.astype(np.float32)
            # normalize the frame
            gray_frame /= 255.0
            # convert the frame to a tensor
            tensor_frame = torch.from_numpy(gray_frame)
            # Run the model
            output = trained_model(tensor_frame)

            # Get the predicted class
            predicted_class = output.data.max(1, keepdim=True)[1].item()

            # print the predicted class
            print(predicted_class)

            # Display the frame
            cv2.imshow('frame',read_frame)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)