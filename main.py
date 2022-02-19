from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset_mnist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()             #调用父类的构造方法
        self.conv1 = nn.Conv2d(3, 32, 3, 1)     #输入1个channel，输出32个channels，kernel_size=3，stride（步长）=1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)    #再变成64channels
        self.dropout1 = nn.Dropout(0.25)        #以0.25的概率dropout
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)         #9216->128
        self.fc2 = nn.Linear(128, 10)

    #定义网络层
    def forward(self, x):
        x = self.conv1(x)
        #线性整流函数（Rectified Linear Unit,ReLU) 是一个激活函数，这是当成一层了
        #卷积神经网络中，若不采用非线性激活，会导致神经网络只能拟合线性可分的数据，因此通常会在卷积操作后，增加非线性激活单元
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #这是两种模式
    #model.train(): 启用BatchNormalization 和 Dropout
    #model.eval():  不器用BatchNormalization 和 Dropout
    #model.eval(),pytorch 会自动把 BN 和 Dropout 固定住，不会取平均，而使用训练好的值
    #不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大；在模型测试阶段使用
    #trainloader对每一个batch加了id
    for batch_idx, (data, target) in enumerate(train_loader):
        #读入数据到device中,之后就用新的变量表示就可,对程序不影响(物理层和应用层)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()       #初始化优化器参数
        output = model(data)
        loss = F.nll_loss(output, target)   #计算loss,此时output已经经过了log和softmax，交叉上核心步骤已经完成
        loss.backward()                     #反向传播
        optimizer.step()                    #调整参数
        #上面的方法都是共享一个参数空间的，所以不需要传递参数
        '''
        if batch_idx % args.log_interval == 0:
            print('batch_idx is {0}, log_interval is {1}'.format(str(batch_idx),str(args.log_interval)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        '''
    print('Train Epoch: {} '.format(epoch),end='\t')

    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()     # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)               # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()   #计算正确的个数，输出正确率

    test_loss /= len(test_loader.dataset)

    print('loss: {:.5f}\t Acc: {:.5f}\n'.format(test_loss,correct / len(test_loader.dataset)))


def main():
    # Training settings
    #都是可选参数，是为了调参用的
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')   #加上参数描述，在--help中输出
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()              #获取参数，从这里就可以开始调用这些参数了。没有输入也没有设置默认值的就是null
    use_cuda = not args.no_cuda and torch.cuda.is_available()   #有cuda且启用时使用cuda

    torch.manual_seed(args.seed)            #设置随机种子，以便于生成随机数
    device = torch.device("cuda" if use_cuda else "cpu")    #选择设备:  CPU or GPU

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    #dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform) #下载训练集
    #dataset2 = datasets.MNIST('../data', train=False,transform=transform)               #下载测试集
    dataset1=dataset_mnist.MyDataset(root='./',datatxt='test_data/label.txt', transform=transforms.ToTensor(),target_transform=None)
    dataset2=dataset_mnist.MyDataset(root='./',datatxt='train_data/label.txt', transform=transforms.ToTensor(),target_transform=None)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    #将模型读入device
    model = Net().to(device)
    #设置优化器，这里使用的是Adagrad优化方法（Adaptive Gradient)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #等间隔调整学习率StepLR
    
    #等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size，单位是step（epoch, not iteration)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):     #迭代次数
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()    #每次迭代之后调整学习率

    if args.save_model:     #保存模型
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
