import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        files = []

        if train:
            files = [os.path.join(base_folder, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            files = [os.path.join(base_folder, "test_batch")]

        X = []
        y = []

        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f, encoding="bytes")
                X.append(data[b"data"])
                y.append(data[b"labels"])

        X = np.concatenate(X)
        y = np.concatenate(y)
        X = X.reshape(-1, 3, 32, 32)
        X = X / 255.0
        self.X = X
        self.y = y


    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        
        return len(self.X)
