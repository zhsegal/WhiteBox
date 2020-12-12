from torch import optim
import pandas as pd

def cache_training_set(trainloader, net, criterion,file_name,nsamples):
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    dataset=pd.DataFrame(columns=['id','grad','weights','loss','prediction','true_label'])
    for (i, inputs) in enumerate(trainloader):

            ## ALEX TRAIN###
        optimizer.zero_grad()
        outputs = net(inputs[0].cuda())
        loss = criterion(outputs[0], inputs[1].cuda())
        loss.backward()
        optimizer.step()

        grads = []
        for param in net.parameters():
            grads.append(param.grad.view(-1))


        id=i
        loss_sample=loss
        true_labes=inputs[1]
        prediction=outputs
        grad=grads[-2]
        weights=net.state_dict()['features.10.weight']
        dataset.loc[i]=[id,grad,weights,loss_sample,prediction,true_labes]
        if i > nsamples:
            break

    dataset.to_csv(f'cache/{file_name}')
    print ('done')
    pass

