from utils import *
import torch
import torch.nn as nn
from test import test
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from nets import alexnet
from white_train import white_train

def train():
    dataset = 'cifar100'
    checkpoint_path = './checkpoints_100cifar_alexnet_white'
    train_batch = 100
    test_batch = 100
    lr = 0.05
    epochs = 500
    state = {}
    state['lr'] = lr
    use_cuda = torch.cuda.is_available()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    print('==> Preparing dataset %s' % dataset)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])

    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    # Model
    print("==> creating model ")
    net = alexnet(num_classes)
    net = torch.nn.DataParallel(net).cuda()
    net = list(net.children())[0]
    net = net.cuda()

    ### TRAIN #####

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    title = 'cifar-100'
    criterion_attack = nn.MSELoss()
    criterion_classifier = nn.CrossEntropyLoss(reduce=False)

    trainset = dataloader(root='./data100', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0)
    testset = dataloader(root='./data100', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=True, num_workers=0)

    epoch = 0
    test_loss, test_acc = white_train(trainloader, net, criterion, epoch, use_cuda)

    test_loss, test_acc = test(testloader, net, criterion, epoch, use_cuda)
    print(test_acc)

    print(f'Hi')

if __name__ == '__main__':
    train()

