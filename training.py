from dataloader import SampleLoader

def train_loop():
    samples_dataset = SampleLoader(csv_file='cache/true_samples.csv')
    print ('dof')