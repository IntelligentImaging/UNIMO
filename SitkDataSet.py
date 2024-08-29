import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import TensorDataset, Dataset
import json

class SitkDataset(Dataset):
    def __init__(self, json_file, keyword, transform=None):
        self.keyword = keyword
        with open(json_file, 'r') as f:
            self.data_info = json.load(f)

    def __len__(self):
        # return 200
        if (self.keyword == "train"):
            return len(self.data_info[self.keyword])
        if (self.keyword == "test"):
            return len(self.data_info[self.keyword])
        if (self.keyword == "all2one"):
            return len(self.data_info[self.keyword])
    def __getitem__(self, idx):
        src = self.data_info[self.keyword][idx]['Image']
        src_img = sitk.ReadImage(src)
        src_data = torch.from_numpy(sitk.GetArrayFromImage(src_img)).unsqueeze(0)
        tag = self.data_info[self.keyword][idx]['Tag']
        return src_data, tag
