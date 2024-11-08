import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms
import torchvision

data_folder = './data'



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop([128,128]),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_data = torchvision.datasets.ImageFolder(
    root='./data/seg_train/seg_train',
    transform = transform, # Specify the transform method in the dataset
)
test_data = torchvision.datasets.ImageFolder(
    root='./data/seg_test/seg_test',
    transform = transform, # Specify the transform method in the dataset
)

############################### output dataset info
print(f"Size of the train set: {len(train_data)}")
classes = train_data.classes
print("Categories and their corresponding labels:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

training_loader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle=True)

epochs = 100

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

model = ConvNet()

if (torch.cuda.is_available()):
    print("CUDA available")
    model = model.cuda()



###############################   output images
# import matplotlib.pyplot as plt
# import numpy as np

# for i in range(1000):
#     sample_data, sample_target = train_data[i]
#     print(f"Size of {i}th image in this dataset: {sample_data.shape}")

# sample_data, sample_target = train_data[0]
# print(f"Size of one image in this dataset: {sample_data.shape}")

# plt.figure(figsize=(11, 4))
# for i in range(3):
#     image, label = test_data[i]
#     image = np.transpose(image, (1, 2, 0))
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(image)
#     plt.axis('off')
# plt.show()




optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_func = torch.nn.CrossEntropyLoss()



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        if (torch.cuda.is_available()):
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def test_after_one_epoch():
    running_loss = 0.
    nonaccurate = 0.
    total = 0.
    for i, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        if (torch.cuda.is_available()):
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        total += len(labels)
        value, index = torch.max(outputs, 1)
        compare = index - labels
        nonaccurate += torch.count_nonzero(compare)
        running_loss += loss.item()
    print('test set: Average loss:{}', format(running_loss / i))
    print('Accuracy:{}', format(1 - nonaccurate/total))





import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir='./log')

test_after_one_epoch()
for epoch in range(epochs):
    print ('epoch {} :', format(epoch))
    train_one_epoch(epoch, writer)
    test_after_one_epoch()

torch.save(model, 'model_100.pth')
