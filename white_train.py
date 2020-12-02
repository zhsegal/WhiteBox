from torch import optim

def white_train(trainloader, net, criterion, epoch, use_cuda):
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    for (i, inputs) in enumerate(trainloader):

            ## ALEX TRAIN###
            optimizer.zero_grad()
            outputs = net(inputs[0].cuda())
            loss = criterion(outputs[0], inputs[1].cuda())
            loss.backward()
            optimizer.step()



            print ('done')
    pass


