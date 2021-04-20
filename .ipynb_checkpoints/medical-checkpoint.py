import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm as tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


train_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/train/",
    transform=transforms.Compose([
        transforms.ToTensor()]
))

val_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/val/",
    transform=transforms.Compose([
        transforms.ToTensor()]
))

test_imgs = torchvision.datasets.ImageFolder(
    "./medical_dataset/test/",
    transform = transforms.Compose([
        transforms.ToTensor()])
)

train_loader = DataLoader(
    train_imgs, batch_size=50, shuffle = True)
val_loader = DataLoader(
    val_imgs, batch_size=50, shuffle=True)
test_loader = DataLoader(
    test_imgs, batch_size=len(test_imgs), shuffle=False)

print(train_imgs[0][0].view(train_imgs[0][0].shape[0], -1))
print(train_imgs[0][0][0])
print(len(train_loader))
print(len(train_imgs))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*53*53, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out

net = CNN().to(device)

writer = SummaryWriter(log_dir='./experiment1')


optimizer=optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)

total_batch = len(train_loader)
num_epochs = 6

train_loss_list=[]
val_loss_list = []
train_acc_list=[]
val_acc_list=[]
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader):
      imgs, labels = data
      imgs = imgs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      out=net(imgs)
      loss = criterion(out, labels)
      loss.backward()
      optimizer.step()
        
      train_loss += loss.item()
      pred = torch.argmax(out, 1) == labels
      train_acc += pred.sum()
        
    
      if (i+1) % 100 ==0:
        writer.add_scalar('training loss',
                            train_loss/100,
                            epoch*len(train_loader)+i)
        writer.close()
        with torch.no_grad():
          val_loss=0.0
          val_acc = 0.0
          for j, val_data in enumerate(val_loader):
            imgs,label = val_data
            imgs = imgs.to(device)
            label = label.to(device)
            val_out = net(imgs)
            v_loss = criterion(val_out, label)
            val_loss += v_loss
            val_pred = torch.argmax(val_out, 1) == label
            val_acc += val_pred.sum()
                    
        print("epoch: {}/{}, step: {}/{}, train loss: {:.4f}, val loss: {:.4f}, train acc: {:.2f}, val acc: {:.2f}".format(
            epoch+1, num_epochs, i+1, total_batch, train_loss/100, val_loss/len(val_loader), train_acc/100/50, val_acc/len(val_loader.dataset)
             ))
            
        train_loss_list.append(train_loss/100)
        val_loss_list.append(val_loss/len(val_loader))
        train_acc_list.append(train_acc/100/50)
        val_acc_list.append(val_acc/len(val_loader.dataset))
        train_loss = 0.0
        train_acc = 0.0


import matplotlib.pyplot as plt

plt.figure(figsize = (16, 9))
x_range = range(len(train_loss_list))
plt.plot(x_range, train_loss_list, label="train")
plt.plot(x_range, val_loss_list, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(figsize = (16, 9))
x_range = range(len(train_loss_list))
plt.plot(x_range, train_acc_list, label="train")
plt.plot(x_range, val_acc_list, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")

with torch.no_grad():
    corr_num = 0
    total_num = 0
    for num, data in enumerate(val_loader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        model_label = prediction.argmax(dim=1)
        
        corr  = label[label == model_label].size(0)
        corr_num += corr 
        total_num += label.size(0)
        
print('Accuracy:{:.2f}'.format(corr_num/total_num*100))

with torch.no_grad():
    for num, data in enumerate(test_loader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        print('Accuracy:{:.2f} %'.format(100*accuracy.item()))