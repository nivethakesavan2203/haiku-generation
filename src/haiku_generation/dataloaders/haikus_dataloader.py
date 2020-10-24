import string

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HaikuDataset(Dataset):
    def __init__(self, filename, is_kaggle=False):
        super(HaikuDataset, self).__init__()

        if not is_kaggle:
            df = pd.read_csv(filename, names=['haiku'])
        else:
            df = pd.read_csv(filename, names=['nan', 'first', 'second', 'third', 'source', 'hash'])
            df['haiku'] = df['first'] + ' ' + df['second'] + ' ' + df['third']
            df['haiku'] = df['haiku'].astype(str)

        df['haiku'] = df['haiku'].apply(lambda x: x.lower())
        df['haiku'] = df['haiku'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df['haiku'] = df['haiku'].apply(lambda x: x.replace('\n', ' '))

    def __len__(self):
        """
        gets the length of the dataset.
        """
        pass

    def __getitem__(self, index: int):
        """
        gets an item from the haikus dataset.
        """
        return 0


if __name__ == '__main__':
    haikus = HaikuDataset('src/datasets/all_haiku.csv', is_kaggle=True)