from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,time
import sys
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tensorboardX import SummaryWriter


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

writer = SummaryWriter()

data_dir = 'train'
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((299,299)),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.RandomRotation((0,360), center=None),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       ])
                                       
    test_transforms = transforms.Compose([transforms.Resize((299,299)),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomRotation((0,360), center=None),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=16)
    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

model = torch.hub.load('pytorch/vision', 'resnext101_32x8d', pretrained=True)
#model = torch.load("resnext.pth").cuda()
model = model.cuda()
#model = nn.DataParallel(model).cuda()
device = ("cuda:0" if torch.cuda.is_available() else "cpu" )

criterion = nn.CrossEntropyLoss()


# for name, child in model.named_children():
#     if name in ['layer3', 'layer4', 'layer2']:
#        print(name + ' is unfrozen')
#        for param in child.parameters():
#            param.requires_grad = True
#     else:
#        print(name + ' is frozen')
#        for param in child.parameters():
#            param.requires_grad = False

#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

epochs = 50
steps = 200
running_loss = 0
print_every = 20
train_losses, test_losses = [], []
for epoch in range(epochs):
    start = time.time()
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            end = time.time()
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader)) 
            writer.add_scalar('Loss/train', running_loss/print_every, steps)
            writer.add_scalar('Loss/test', test_loss/len(testloader), steps)
            writer.add_scalar('Accuracy/train', accuracy/len(testloader), steps)
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
    torch.save(model, 'resnext.pth')
writer.close()
