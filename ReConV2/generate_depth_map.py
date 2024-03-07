from PIL import Image
from ReConV2.utils.misc import *
from ReConV2.utils.config import *
from ReConV2.datasets import build_dataset_from_cfg
from ReConV2.datasets.pc_render import Realistic_Projection

dataset = 'hybrid'
save_path = f'ReConV2/data/{dataset}/depth/'

origin_config = cfg_from_yaml_file(f'ReConV2/cfgs/pretrain/{dataset}.yaml')
config = origin_config.dataset.train
config.others.img_views = 10
dataset = build_dataset_from_cfg(config._base_, config.others)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               num_workers=8,
                                               drop_last=False,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True)

pc_views = Realistic_Projection()
get_img = pc_views.get_img


def real_proj(pc, imsize=256):
    img = get_img(pc.unsqueeze(0))
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)
    return img


for idx, (pts, _, _, index) in enumerate(train_dataloader):
    print(idx)
    pts = pts.cuda()
    img = real_proj(pts)
    B, n, c, w, h = img.shape

    for i in range(B):
        for j in range(n):
            tensor_image = (img[i, j].numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np.transpose(tensor_image, (1, 2, 0)))
            if dataset == 'hybrid':
                path = save_path + index[i].replace("/", "-")[:-4] + f'-{j}'
            else:
                path = save_path + index[i] + f'-{j}'
            pil_image.save(path)
