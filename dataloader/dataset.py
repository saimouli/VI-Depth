import pytorch_lightning as pl
from typing import Tuple
from torch.utils.data import DataLoader
import os
from data.SML_dataset import SML_dataset
from data.SML_consistent_dataset import SML_consistent_dataset
from data.SML_tartan_consistent_dataset import SML_tartan_consistent_dataset

class SMLDataConsistentModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str = "",
            sequence_length: int = 3,
            img_size: Tuple = (480, 640),
            batch_size: int = 12,
            num_workers: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.data_root = data_root
        #self.data_train = os.path.join(data_root, 'training')
        #self.data_val = os.path.join(data_root, 'testing')

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SML_tartan_consistent_dataset( #SML_consistent_dataset(
                self.data_root,
                mode="train",
                sequence_length=self.sequence_length
            )
            self.test_dataset = SML_tartan_consistent_dataset( #SML_consistent_dataset(
                self.data_root,
                mode="val",
                sequence_length=self.sequence_length
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
 
class SMLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str = "",
            img_size: Tuple = (480, 640),
            batch_size: int = 12,
            num_workers: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train = os.path.join(data_root, 'training')
        self.data_val = os.path.join(data_root, 'testing')

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SML_dataset(
                self.data_train,
                mode="train",
                depth_scale=256.0
            )
        
        if stage in ("fit", "validate"):
            self.test_dataset = SML_dataset(
                self.data_val,
                mode="val",
                depth_scale=256.0
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
