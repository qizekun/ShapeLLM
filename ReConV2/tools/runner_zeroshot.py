import torch
import torch.nn as nn
from ReConV2.tools import builder
from ReConV2.utils.logger import *
from ReConV2.tools.runner_pretrain import Trainer
from ReConV2.datasets import data


def run_net(args, config):
    logger = get_logger(args.log_name)

    # build model
    device = torch.device("cuda", args.local_rank)

    base_model = builder.model_builder(config.model)
    base_model.to(device)

    modelnet40_loader = data.make_modelnet40test(config)
    scanobjectnn_loader = data.make_scanobjectnntest(config)
    objaverse_lvis_loader = data.make_objaverse_lvis(config)

    base_model.zero_grad()
    triner = Trainer(args.local_rank, args, config, base_model, None, None, device,
                     logger, modelnet40_loader=modelnet40_loader, scanobjectnn_loader=scanobjectnn_loader,
                     objaverse_lvis_loader=objaverse_lvis_loader)

    triner.load_from_checkpoint(args.ckpts)
    triner.model_parallel()
    triner.test_modelnet40()
    triner.test_scanobjectnn()
    triner.test_objaverse_lvis()
