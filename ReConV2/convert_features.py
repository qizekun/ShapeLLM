from utils.misc import *
from utils.config import *
from datasets import build_dataset_from_cfg
from models.CrossModal import VisionTransformer as ImageEncoder

origin_config = cfg_from_yaml_file('ReConV2/cfgs/pretrain/base/cap3d.yaml')
config = origin_config.dataset.train
config.others.img_views = 8
dataset = build_dataset_from_cfg(config._base_, config.others)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               num_workers=8,
                                               drop_last=False,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True)

result = {}
img_encoder = ImageEncoder(origin_config.model).cuda()

for idx, (_, img, _, index) in enumerate(train_dataloader):
    print(idx)
    B, n, c, w, h = img.shape
    img = img.cuda()
    img = img.reshape(B * n, c, w, h)
    feature = img_encoder(img)
    feature = feature.reshape(B, n, -1)
    for i in range(B):
        torch.save(feature[i].cpu(), f'/data/features/cap3d/{index[i]}.pt')
