import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# you need to initalize the class again
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 hidden layers with 32 / 64 / 128 neurons . this caculates in convolution 2d
        self.conv1 = nn.Conv2d(1,32,5) # 1 input , 32 neurons , 5 - kernel size
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)

        x = torch.randn(50,50).view(-1,1,50,50) # we need to convert 2d to 1d so we need to find the 1d size
        self._to_linear = None # using random generated data to find the size
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512) # calculation in 1-d using 512 neurons
        self.fc2 = nn.Linear(512,2) # out put 2 neurons cat / dog

    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))#using pooling and activation function to round off values
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

        #print(x[0].shape)

        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # to get the size of 1-d or flattened img

        return x

    def forward (self,x):
        x = self.convs(x) # calculating convolution first
        x = x.view(-1,self._to_linear) # converting to linear
        x = F.relu(self.fc1(x)) # calculating linear
        x = self.fc2(x) # getting output
        return F.softmax(x,dim =1) #using activation function at output to get % or 0-1 values

net = Net()
#loading the saved parameters
save_path = os.path.join("model.pt")
net.load_state_dict(torch.load(save_path))
net.eval()

# To check if a random image is a dog or a cat
while True:

    get_path = input("Enter the path of the image :")

    #save_path = os.path.join("Enter image name")
    #if the image is in ur code folder use the above code

    img = cv2.imread(get_path)

    X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X = cv2.resize(X, (50,50))

    X = torch.Tensor(np.array(X)).view(-1,50,50)
    #gets all the image values from dataset , in the size 50x50
    X = X/255.0
    # since gray scale is of pixels from 0-255 converting to 0-1

    cod = net((X.view(-1,1,50,50)))
    check_cod = torch.argmax(cod)
    print(cod,check_cod)
    if check_cod == 0:
        animal = "Cat"

    else :
        animal = "Dog"

    plt.axis("off")
    plt.title(animal)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    yorn = input("Do you want to check for another image (y/n) ?")

    if yorn == "n" or yorn == "N" :
        break



