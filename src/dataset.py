# ========================DATA_SET=============================== 

import pandas as pd
from torch.utils.data import Dataset

class WebAttackDataSet(Dataset):
    """
    Dataset class for web attack analysis. 
    Every dataset using pytorch should be overwrite this class
    This require 2 function, __len__ and __getitem__
    """
    def __init__(self, data_file, text_column, label_column):
        """
        Args:
            data_file (string): path to data file
        """
        self.df = pd.read_csv(data_file)
        self.df.reset_index(drop=True, inplace=True)
        self.text_column = text_column
        self.label_column = label_column
        
        # print(self.df.columns)
        # print(text_column)
        # print(label_column)

    def __len__(self):
        """
        length of the dataset, i.e. number of rows in the csv file
        Returns: int 
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        given a row index, returns the corresponding row of the csv file
        Returns: text (string), label (int) 
        """
        text = self.df.iloc[idx][self.text_column]
        label = self.df.iloc[idx][self.label_column]
        return text, label