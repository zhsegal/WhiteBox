from dataloader import SampleLoader
from torch.utils.data import Dataset, DataLoader
from nets import InferenceAttack

def train_loop():
    batch_size=4
    samples_dataset = SampleLoader(file_name='cache/true_samples.pt')
    dataloader = DataLoader(samples_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    net=InferenceAttack(100)

    for batch in (dataloader):
        ids=batch['id']
        grad = batch['grad']
        loss = batch['loss']
        prediction = batch['prediction']
        true_label = batch['true_label']
        weights = batch['weights']

        outopus=net(grad.cuda(), loss.cuda(), prediction[0].cuda(),true_label.cuda())


        print (batch)

    print ('dof')