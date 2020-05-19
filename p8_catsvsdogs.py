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

#make sure to extract kaggle PetTmages folder in the same file as your code

REBUILD_DATA = False # True - to build a dataset , false - to not build a dataset

class DogsVSCats():
    IMG_SIZE = 50 # making the img 50x50 pixels
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)): #runs till the last img in the directory
                if "jpg" in f: #getting only jpg
                    try: #used for error handling because there are some corrupt imgs
                        path = os.path.join(label, f) 
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)#converting to gray scale to reduce complexity
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) 
                        # converts the img to 50x50 px
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) 
                        # assigns [1,0] as cat and [0,1] as dog (using one hot vector) 

                        if label == self.CATS: #used to check balance of inputs b/w cats and dogs
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data) #shuffling the cats and dogs data for efficient generalization
        np.save("training_data.npy",self.training_data) #saving it
        print("Cats :",self.catcount) #checking the count
        print("Dogs :",self.dogcount)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #no of channels used is 1
        # 3 hidden layers with 32 / 64 / 128 neurons . this caculates in convolution 2d
        self.conv1 = nn.Conv2d(1,32,5) # 1 channel , 32 neurons , 5 - kernel size
        self.conv2 = nn.Conv2d(32,64,5) 
        self.conv3 = nn.Conv2d(64,128,5)

        x = torch.randn(50,50).view(-1,1,50,50) # we need to convert 2d to 1d so we need to find the 1d size
        self._to_linear = None # using random generated data to find the size
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512) # hidden layer with 512 neurons  
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


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    #To check if the ouput matches the labels
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    #calculating accuracy and loss %
    if train:
        loss.backward()
        optimizer.step()
        #reducing loss and increasing accuracy

    return acc, loss

#To test if the neural network is trained correctly by using the images validated
def test(size = 32):
    random_start = np.random.randint(len(test_X)-size) #getting a random slice of given size
    X,y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad(): #calculating acc and loss for the test images
        val_acc , val_loss = fwd_pass(X.view(-1, 1,50,50).to(device) , y.to(device))
    return val_acc, val_loss

def train():
    BATCH_SIZE = 100 #no of samples in a go
    EPOCHS = 8 # no of full passes
    with open("model_graph.log","a") as f: #to register acc,loss of test and train images for plotting graph
        for epoch in range(EPOCHS):
            
            for i in tqdm(range(0,len(train_X),BATCH_SIZE)): #there are around 25500(approx) images 
                #so the loop runs for 25500/100 times that is 255 times 
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)
                acc,loss = fwd_pass(batch_X , batch_y , train= True)
                #getting acc and loss of train images
                if i % 50 == 0: #updating loss and acc after every 50 iterations
                    val_acc , val_loss = test(size = 100) #getting acc and loss of test images after training 50 times
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")



# Main program    

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")

else:
    device = torch.device("cpu")
    print("running on the CPU")

# If gpu is avaliable it will do calculations on the GPU

net = Net().to(device)
#calling the neural network

if REBUILD_DATA: #To build the data REBUILD_DATA = True
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()


training_data = np.load("training_data.npy",allow_pickle=True) # loading data set

#setting up inputs / optimizer / loss functions etcc... 


optimizer = optim.Adam(net.parameters(), lr=0.001)
#using optimizer to tune parameters and learning rate 0.001
loss_function = nn.MSELoss()
#using mean square error to calculate loss ( because one hot vector)

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
#converting numpy array to tensor 
X = X/255.0
# since gray scale is of pixels from 0-255 converting to 0-1
y = torch.Tensor([i[1] for i in training_data])
#Getting the labels for corresponding image values
VAL_PCT = 0.1  # lets reserve 10% of images for validation (test images)
val_size = int(len(X)*VAL_PCT)
#print(val_size)

train_X = X[:-val_size] #gets all images other than test images
train_y = y[:-val_size]

test_X = X[-val_size:] #gets all test images
test_y = y[-val_size:]


MODEL_NAME = f"model -{int (time.time())}"
print(MODEL_NAME)
#giving our model a name

train() #to train

save_path = os.path.join("model.pt")
torch.save(net.state_dict(),save_path)
#saving parameters