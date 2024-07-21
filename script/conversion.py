import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
# network structure
import torch.nn as nn
import torch.nn.functional as F
import csv

# Provide the DNN model here, we use LeNet-5 as an example
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # load model file and trained weight file
    net = LeNet5()
    net.load_state_dict(torch.load('./models/lenet_mnist.pt'))

    ### example script of weight conversion as NoCDAS input file
    f = open("./models/weight.txt", "a", newline='')
    for layerNum, layer in enumerate(net.children()):
        # if layerNum <= 2:
        #     continue
        print(layerNum, layer)
        if isinstance(layer, nn.Conv2d):
            w = layer.weight.data.detach().numpy()
            b = layer.bias.data.detach().numpy()
            n = b.shape[0]
            w = w.reshape((n, -1))
            b = b.reshape((n, -1))
            w = np.concatenate((w, b), axis=1)
            csv.writer(f, delimiter=' ').writerows(w)
        elif isinstance(layer, nn.Linear):
            w = layer.weight.data.detach().numpy()
            b = layer.bias.data.detach().numpy()
            n = b.shape[0]
            b = b.reshape((n, -1))
            w = np.concatenate((w, b), axis=1)
            csv.writer(f, delimiter=' ').writerows(w)
    
    f.close()

    ### example script of input conversion as NoCDAS input file
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    f = open("./models/input.txt", "a", newline='')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0], data[1]
            i = images.detach().numpy()
            i = i.reshape((28, 28))
            y = np.pad(i, (2, 2), 'constant')
            csv.writer(f, delimiter=' ').writerows(y)
            #print(y)
            break

    f.close()
