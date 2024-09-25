from PIL import Image
from tqdm import tqdm
from ReConV2.utils.misc import *
from ReConV2.datasets.HybridDataset import Hybrid_points
from ReConV2.datasets.pc_render import Realistic_Projection

data_root = 'ReConV2/data/HybridDatasets/'
save_path = 'ReConV2/data/HybridDatasets/depth/'
batch_size = 32

if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = Hybrid_points(data_root, 'train', 1024)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=8,
                                               drop_last=False,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True)

pc_views = Realistic_Projection()
get_img = pc_views.get_img


def real_proj(pc, imsize=256):
    img = get_img(pc)
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)
    return img


for pts, index in tqdm(train_dataloader):
    pts = pts.cuda()
    img = real_proj(pts)
    n, c, w, h = img.shape
    batch_size = n // 10
    img = img.reshape(batch_size, 10, c, w, h)

    for i in range(batch_size):
        for j in range(10):
            tensor_image = (img[i, j].cpu().detach().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np.transpose(tensor_image, (1, 2, 0)))
            path = save_path + index[i].replace("/", "-")[:-4] + f'-{j}.png'
            pil_image.save(path)
