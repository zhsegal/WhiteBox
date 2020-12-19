from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os

class SampleLoader(Dataset):

    def __init__(self, file_name):
        self.pt_file=file_name

    def __len__(self):
        return len(self.pt_file)

    def __getitem__(self, idx):


        torch_file = self.pt_file
        df=torch.load(torch_file)
        sample_line = df[idx]

        id_num=sample_line[0]
        grad=sample_line[4]
        weights=sample_line[5]
        loss=sample_line[1]
        prediction=sample_line[3]
        true_label=sample_line[2]


        sample = {'id': id_num, 'grad': grad,'weights': weights,'loss': loss, 'prediction': prediction,'true_label': true_label}

        return sample