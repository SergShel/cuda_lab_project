import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
# need ms coco api to get dataset
from pycocotools import mask as mask
import numpy as np
import matplotlib.pyplot as plt
# "PYTHONPATH=. python .py" to import when running
import utils.utils
import utils.train_eval

import datasets.cityscapes_loader as cityscapes_loader
import architectures.Temporal_UNET_Template as Temporal_UNET_Template
from architectures.architecture_configs import *

if __name__ == "__main__":
    utils.utils.set_random_seed()

    is_sequence = False

    dataset_root_dir = "/home/nfs/inf6/data/datasets/cityscapes/"

    train_ds = cityscapes_loader.cityscapesLoader(
        root=dataset_root_dir, split='train', img_size=(512, 1024), is_transform=True, is_sequence=is_sequence
        )
    val_ds = cityscapes_loader.cityscapesLoader(
        root=dataset_root_dir, split='val', img_size=(512, 1024), is_transform=True, is_sequence=is_sequence
        )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=True)

    encoder_blocks = SmallShallow_NetworkSize.encoder_blocks
    decoder_blocks = SmallShallow_NetworkSize.decoder_blocks


    config = Temporal_ConvUNextConfig(
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        temporal_cell= Conv2dGRUCell
        )

    temp_unet = Temporal_UNET_Template.Temporal_UNet(config)

    temp_unet_optim = torch.optim.Adam(temp_unet.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()


    epochs=85
    temp_unet_trainer = utils.train_eval.Trainer(
            temp_unet, temp_unet_optim, criterion,
            train_loader, valid_loader, "cityscapes", epochs,
            sequence=is_sequence, all_labels=20, start_epoch=0)

    # temp_unet_trainer.save_model(0)

    # load_model = True
    # if load_model:
    #     temp_unet_trainer.load_model("cityscapes")


    temp_unet_trainer.train_model()
