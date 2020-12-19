import torch
import torch.nn as nn

import torch.nn.parallel


class InferenceAttack(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(InferenceAttack, self).__init__()
        self.grads_conv = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=1),
            nn.ReLU(),

        )
        self.grads_linear = nn.Sequential(

            nn.Dropout(p=0.2),
            nn.Linear(256 * 100, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.loss = nn.Sequential(
            nn.Linear(1, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.preds = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        self.correct = nn.Sequential(
            nn.Linear(1, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * 4, 256),

            nn.ReLU(),
            nn.Linear(256, 128),

            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            print(key)
            if key.split('.')[-1] == 'weight':
                nn.init.normal(self.state_dict()[key], std=0.01)
                print(key)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, grad, loss, pred, label):
        out_g = self.grads_linear(grad.view([grad.size()[0], -1]))
        out_l = self.loss(loss)
        out_c = self.correct(label)
        out_o = self.preds(pred)

        is_member = self.combine(torch.cat((out_g, out_c, out_l, out_o), 1))

        return self.output(is_member)

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inp):
        outputs = []
        x = inp
        module_list = list(self.features.modules())[1:]
        for l in module_list:
            x = l(x)
            outputs.append(x)

        y = x.view(inp.size(0), -1)
        o = self.classifier(y)
        return o, outputs[-1].view(inp.size(0), -1), outputs[-4].view(inp.size(0), -1)


def alexnet(classes):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(classes)
    return model


# In[75]:


class InferenceAttack_HZ(nn.Module):
    def __init__(self, num_classes, num_layers=1):
        self.num_layers = num_layers
        self.num_classes = num_classes
        super(InferenceAttack_HZ, self).__init__()

        self.grads_conv = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 100, kernel_size=(1, 100), stride=1),
            nn.ReLU(),

        )
        self.grads_linear = nn.Sequential(

            nn.Dropout(p=0.2),
            nn.Linear(256 * 100, 2024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.preds = nn.Sequential(
            nn.Linear(num_classes, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        self.preds2 = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        self.preds3 = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
        )
        self.correct = nn.Sequential(
            nn.Linear(1, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * (6 + num_layers), 256),

            nn.ReLU(),
            nn.Linear(256, 128),

            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            print(key)
            if key.split('.')[-1] == 'weight':
                nn.init.normal(self.state_dict()[key], std=0.01)
                print(key)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, g, l, c, o, l1, l2):
        out_g = self.grads_conv(g).view([g.size()[0], -1])
        out_g = self.grads_linear(out_g)
        out_l = self.labels(l)
        out_c = self.correct(c)
        out_o = self.preds(o)
        # out_g1 = self.preds2(l1)
        # out_g2 = self.preds3(l2)

        _outs = torch.cat((out_g, out_c, out_l), 1)

        if self.num_layers > 0:
            _outs = torch.cat((_outs, out_o), 1)
        #         if self.num_layers>1:
        #             _outs= torch.cat((_outs,out_l1),1)

        #         if self.num_layers>2:
        #             _outs= torch.cat((_outs,out_l2),1)

        is_member = self.combine(_outs)

        return self.output(is_member)

