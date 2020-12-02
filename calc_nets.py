## PSEUDO CODE####

import pandas as pd

def calc_nets(train_loader, net, optimizer, criterion):

    nets_df=pd.DataFrame()

    for train_sample in train_loader:
        optimizer.zero_grad()
        outputs = net(train_sample[0].cuda())
        loss = criterion(outputs[0], train_sample[1].cuda())
        loss.backward()
        optimizer.step()

        grad=net.grads
        outputs=net.outputs
        preds=outputs

        temp_df=pd.DataFrame({'grad':grad, 'outputs':outputs, 'preds': preds})
        nets_df.append(temp_df)

    return nets_df