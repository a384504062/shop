import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torchvision.datasets as data
import torchvision
import MLP_net

def Trainer():
    dataloader = data.MNIST('MNIST',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    dataloadertest = data.MNIST('MNIST',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    datas = data_utils.DataLoader(dataset=dataloader,batch_size=512,shuffle=True)
    datastest = data_utils.DataLoader(dataset=dataloadertest,batch_size=512,shuffle=True)
    nett = MLP_net.Net().cuda()
    loss_fun = torch.nn.MSELoss()
    optimer = optim.Adam(nett.parameters(),lr=0.01)
    for i in range(10):
        print(i,'轮')
        for a,(b,c) in enumerate(datas):
            b = b.cuda()
            b = torch.reshape(b, shape=(-1,28*28*1))
            nets = nett(b).cuda()
            c = torch.zeros(c.size(0),10).scatter_(1,c.view(-1,1),1).cuda()
            loss = loss_fun(nets,c)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
        print("loss:",loss.item())
        with torch.no_grad():
            nett.eval()
            total = 0
            num_correct = 0
            for d,e in datastest:
                d = d.cuda()
                d = torch.reshape(d,shape=(-1,28*28*1))
                nettout = nett(d).cuda()
                _, predicted = torch.max(nettout.data, 1)
                e = e.cuda()
                total += e.size(0)
                num_correct += (predicted == e).sum()
                accuracy = num_correct.float()/total
            print("accuracy:{0}%".format(100*accuracy))
            torch.save(nett,'MLP_net.pth')
if __name__ == '__main__':
    trainer = Trainer()