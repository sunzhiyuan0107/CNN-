import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms
import torchvision

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # Set for convolution operation
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4,4)
        )
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.conv3=torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )

        self.conv4=torch.nn.Sequential(
            torch.nn.Conv2d(64,64,3,padding=1),
            torch.nn.ReLU(),
        )

        self.fc1=torch.nn.Sequential(
            torch.nn.Linear(64*8*8,32),
            torch.nn.ReLU(),
        )
        self.fc2=torch.nn.Linear(32,10)



    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,64*8*8)
        x = self.fc1(x) 
        x = self.fc2(x) 
        out=F.log_softmax(x,dim=1)
        return out
model = torch.load('model_100.pth')
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop([128,128]),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

predict = torchvision.datasets.ImageFolder(
    root='./data/seg_pred',
    transform = transform, # Specify the transform method in the dataset
)

predict_raw = torchvision.datasets.ImageFolder(
    root='./data/seg_pred',
    
)

train_data = torchvision.datasets.ImageFolder(
    root='./data/seg_train/seg_train',
    transform = transform, # Specify the transform method in the dataset
)

print(f"Size of the train set: {len(train_data)}")
classes = train_data.classes
print("Categories and their corresponding labels:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

start_num = 32
for i in range(16):
    image, label = predict[i + start_num]
    image_raw, label_raw = predict_raw[i + start_num]
    image_copy = image
    image_copy = image_copy.cuda()
    outcome = model(image_copy)
    value, index = torch.max(outcome, 1)

    image = np.transpose(image, (1, 2, 0))
    plt.subplot(4, 4, i + 1)
    plt.imshow(image_raw)
    plt.axis('off')
    for i, class_name in enumerate(classes):
        if index == i:
            plt.title(class_name)
    
plt.show()