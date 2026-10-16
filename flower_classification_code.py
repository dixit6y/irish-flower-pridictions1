# Extracted code from flower-classification notebook
# Note: This is the notebook code combined in its original order.


# ============ Import necessary libraries ============

# --- Original code cell 2 ---
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# --- Original code cell 3 ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# --- Original code cell 4 ---
transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(0.3),
                                transforms.ToTensor(),
                                transforms.Resize((80,80))
                               ])

dataset = torchvision.datasets.ImageFolder(root = '../input/flowers-recognition/flowers/flowers',
                                           transform = transform)
print("No of Classes: ", len(dataset.classes))

train, val = torch.utils.data.random_split(dataset, [3000, 1323])

train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = 32, 
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader(dataset = val,
                                         batch_size = 32, 
                                         shuffle = True)

# --- Original code cell 5 ---
examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i].numpy().transpose())
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

# --- Original code cell 6 ---
Accuracies = []

# ============ Custom CNN ============

# --- Original code cell 8 ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(2*2*1024, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 5)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        out = F.dropout(out, training=self.training)
        out = self.fc3(out)
        return F.log_softmax(out,dim=1)

# --- Original code cell 9 ---
model = ConvNet().to(device)

# ============ Loss and Optimizer ============

# --- Original code cell 11 ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============ Model Training ============

# --- Original code cell 13 ---
total_step = len(train_loader)
Loss = []
Acc = []
Val_Loss = []
Val_Acc = []
n=15


for epoch in range(n):
  acc = 0
  val_acc = 0
  for i, (images, labels) in enumerate(train_loader):
    model.train()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  acc = acc/len(train_loader.dataset) * 100
    
  for i, (images, labels) in enumerate(val_loader):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(images)
    val_loss = criterion(outputs, labels)
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    val_acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  val_acc = val_acc/len(val_loader.dataset) * 100
    
  print("Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   Val_loss : {val_loss:.2f};   Val_Accuracy : {val_acc:.2f}%".format(epoch+1, loss=loss.item(), acc=acc, val_loss=val_loss.item(), val_acc=val_acc))
  
  Loss.append(loss)
  Acc.append(acc)

  Val_Loss.append(val_loss)
  Val_Acc.append(val_acc)

# --- Original code cell 14 ---
plt.plot(range(n),Loss)
plt.plot(range(n),Val_Loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

# --- Original code cell 15 ---
plt.plot(range(n),Acc)
plt.plot(range(n),Val_Acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()

# ============ Model Evaluation ============

# --- Original code cell 17 ---
model.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        y_pred = []
        Images = images.to(device)
        Labels = labels.to(device)
        outputs = model(Images)
        prediction_array = outputs.data
        
        _, predicted = torch.max(prediction_array, 1)
        y_pred += predicted
        total += Labels.size(0)
        correct += (predicted == Labels).sum().item()
        
    acc = 100 * correct / total
    Accuracies.append(acc)
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# ============ Transfer Learning ============

# --- Original code cell 19 ---
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(0.2),
                                transforms.ToTensor(),
                                transforms.Resize((224,224))
                               ])

dataset = torchvision.datasets.ImageFolder(root = '../input/flowers-recognition/flowers/flowers',
                                           transform = transform)
print("No of Classes: ", len(dataset.classes))

train, val = torch.utils.data.random_split(dataset, [3000, 1323])

train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = 32, 
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader(dataset = val,
                                         batch_size = 32, 
                                         shuffle = True)

# ============ VGG19 ============

# --- Original code cell 21 ---
vgg = torchvision.models.vgg19(pretrained=True)

# --- Original code cell 22 ---
vgg

# --- Original code cell 23 ---
vgg.classifier[6].out_features = 5
for param in vgg.features.parameters(): 
    param.requires_grad = False

vgg = vgg.cuda()

# ============ Loss and Optimizer ============

# --- Original code cell 25 ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters(), lr=0.001)

# ============ Training Model ============

# --- Original code cell 27 ---
total_step = len(train_loader)
Loss = []
Acc = []
Val_Loss = []
Val_Acc = []


for epoch in range(n):
  acc = 0
  val_acc = 0
  for i, (images, labels) in enumerate(train_loader):
    vgg.train()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = vgg(images)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  acc = acc/len(train_loader.dataset) * 100
    
  for i, (images, labels) in enumerate(val_loader):
    vgg.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = vgg(images)
    val_loss = criterion(outputs, labels)
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    val_acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  val_acc = val_acc/len(val_loader.dataset) * 100
    
  print("Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   Val_loss : {val_loss:.2f};   Val_Accuracy : {val_acc:.2f}%".format(epoch+1, loss=loss.item(), acc=acc, val_loss=val_loss.item(), val_acc=val_acc))
  
  Loss.append(loss)
  Acc.append(acc)

  Val_Loss.append(val_loss)
  Val_Acc.append(val_acc)

# --- Original code cell 28 ---
plt.plot(range(n),Loss)
plt.plot(range(n),Val_Loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

# --- Original code cell 29 ---
plt.plot(range(n),Acc)
plt.plot(range(n),Val_Acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()

# ============ Model Evaluation ============

# --- Original code cell 31 ---
vgg.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        y_pred = []
        Images = images.to(device)
        Labels = labels.to(device)
        outputs = vgg(Images)
        prediction_array = outputs.data
        
        _, predicted = torch.max(prediction_array, 1)
        y_pred += predicted
        total += Labels.size(0)
        correct += (predicted == Labels).sum().item()

    acc = 100 * correct / total
    Accuracies.append(acc)
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# ============ ResNet18 ============

# --- Original code cell 33 ---
resnet = torchvision.models.resnet18(pretrained=True)

# --- Original code cell 34 ---
resnet

# --- Original code cell 35 ---
ftr = resnet.fc.in_features
resnet.fc = nn.Linear(ftr, 5)

resnet = resnet.cuda()

# ============ Loss and optimizer ============

# --- Original code cell 37 ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

# ============ Training model ============

# --- Original code cell 39 ---
total_step = len(train_loader)
Loss = []
Acc = []
Val_Loss = []
Val_Acc = []
n=15

for epoch in range(n):
  acc = 0
  val_acc = 0
  for i, (images, labels) in enumerate(train_loader):
    resnet.train()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = resnet(images)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  acc = acc/len(train_loader.dataset) * 100
    
  for i, (images, labels) in enumerate(val_loader):
    resnet.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = resnet(images)
    val_loss = criterion(outputs, labels)
    
    # Checking accuracy
    preds = outputs.data.max(dim=1,keepdim=True)[1]
    val_acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
  val_acc = val_acc/len(val_loader.dataset) * 100
    
  print("Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   Val_loss : {val_loss:.2f};   Val_Accuracy : {val_acc:.2f}%".format(epoch+1, loss=loss.item(), acc=acc, val_loss=val_loss.item(), val_acc=val_acc))
  
  Loss.append(loss)
  Acc.append(acc)

  Val_Loss.append(val_loss)
  Val_Acc.append(val_acc)

# --- Original code cell 40 ---
plt.plot(range(n),Loss)
plt.plot(range(n),Val_Loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

# --- Original code cell 41 ---
plt.plot(range(n),Acc)
plt.plot(range(n),Val_Acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()

# ============ Model Evaluation ============

# --- Original code cell 43 ---
model.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        y_pred = []
        Images = images.to(device)
        Labels = labels.to(device)
        outputs = resnet(Images)
        prediction_array = outputs.data
        
        _, predicted = torch.max(prediction_array, 1)
        y_pred += predicted
        total += Labels.size(0)
        correct += (predicted == Labels).sum().item()

    acc = 100 * correct / total
    Accuracies.append(acc)
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# ============ Comparison of accuracies of different models ============

# --- Original code cell 45 ---
plt.plot(range(3), Accuracies, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12) 
plt.ylabel('Acc')
plt.xlabel('Models')
plt.title("Accuracies")
plt.xticks(range(3), ['Custom CNN', 'Vgg19', 'Resnet18'])
plt.show()
