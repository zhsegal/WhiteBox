from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os

class SampleLoader(Dataset):

    def __init__(self, csv_file):
        self.csv_file=csv_file

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        csv_path = self.csv_file
        df=pd.read_csv(csv_path)
        sample_line = df.iloc[idx]

        id_num=sample_line['id']
        grad=sample_line['grad']
        weights=sample_line['weights']
        loss=sample_line['loss']
        prediction=sample_line['prediction']
        true_label=sample_line['true_label']


        sample = {'id': id_num, 'grad': grad,'weights': weights,'loss': loss, 'prediction': prediction,'true_label': true_label}

        return sample