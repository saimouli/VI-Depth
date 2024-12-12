from utils_eval import compute_ls_solution
from data.SML_dataset import SML_dataset
import os
import numpy as np
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from model.main import midasNetModule
from model.main_consistent import midasNetConsistentModule


def cli_main():
    #cli = LightningCLI(midasNetModule, datamodule_class=None)
    cli = LightningCLI(midasNetConsistentModule, datamodule_class=None)

if __name__ == "__main__":
    cli_main()
    # train_root = '/media/saimouli/Data6T/datasets/VOID_150/training'
    # result_root = '/media/saimouli/Data6T/datasets/VOID_150/results'
    # sml_ckpt_pt = None

    # image_path = os.path.join(train_root, 'image')
    # gt_path = os.path.join(train_root, 'ground_truth')
    # sparse_depth_path = os.path.join(train_root, 'sparse_depth')
    # DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

