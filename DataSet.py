import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import json
import numpy as np


# 设置data set
class MyData(TensorDataset):
    def __init__(self, data, *tensors: Tensor):
        super().__init__(*tensors)
        self.data = data

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]

    def __len__(self):
        return len(self.data)
