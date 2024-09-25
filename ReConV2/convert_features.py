from utils.misc import *
from utils.config import *
from datasets.HybridDataset import Hybrid_depth

import timm
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = timm.create_model("vit_gigantic_patch14_clip_224.laion2b", pretrained=True).to(device)

data_root = 'ReConV2/data/HybridDatasets/'
img_path = 'ReConV2/data/HybridDatasets/depth/'
save_path = 'ReConV2/data/HybridDatasets/depth_feature/'
batch_size = 32

if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = Hybrid_depth(data_root, 'train', img_path)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=8,
                                               drop_last=False,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True)

for img, id in tqdm(train_dataloader):
    B, n, h, w, c = img.shape
    img = img.reshape(B * n, h, w, c)
    img = img.to(device)
    feature = clip_model(img)
    feature = feature.reshape(B, n, -1)
    for i in range(B):
        torch.save(feature[i].cpu(), save_path + id[i] + '.pt')
