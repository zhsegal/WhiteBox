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
from train_cache import cache_training_set
from training import train_loop

def train():
    dataset = 'cifar100'
    checkpoint_path = './checkpoints_100cifar_alexnet_white'
    train_batch = 1
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

    cacheing=False
    if cacheing:
        cache_training_set(trainloader, net, criterion, 'true_samples.csv',10000)
        cache_training_set(testloader, net, criterion, 'false_samples.csv',2000)

    train_loop()

    # test_loss, test_acc = test(testloader, net, criterion, epoch, use_cuda)

    print(f'Hi')

if __name__ == '__main__':
    train()

