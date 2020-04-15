import torch.nn as nn
import torch
import PIL.Image as Image
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28*1, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 58), nn.ReLU(), nn.BatchNorm1d(58),
            nn.Linear(58, 96), nn.ReLU(), nn.BatchNorm1d(96),
            nn.Linear(96, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 10), nn.ReLU(), nn.BatchNorm1d(10),nn.Softmax()
        )
    def forward(self, x):
        out = self.layers(x)
        return out

if __name__ == '__main__':
    net = Net()
    path = r'0.jpg'
    image = Image.open(path)
    image = np.array(image)
    image = torch.Tensor(image)
    image = image.permute(2,0,1)
    image = image.unsqueeze_(0)
    image = torch.reshape(image, shape=(-1,28*28*1))
    a = net(image)
    print(a,a.shape)







