import torch
import numpy as np
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

from transformers import StoppingCriteria
from llava.constants import POINT_TOKEN_INDEX


def interpolate_point_cloud(points, k):
    N, _ = points.shape
    nn = NearestNeighbors(n_neighbors=k * k, algorithm='ball_tree').fit(points[:, :3])
    distances, indices = nn.kneighbors(points[:, :3])

    interpolated_points = []
    for i in range(N):
        neighbors_indices = indices[i, 1:]
        neighbors = points[neighbors_indices]
        selected_points = np.random.choice(neighbors.shape[0], k, replace=False)
        neighbors = neighbors[selected_points]

        interpolated_point = (points[i] + neighbors) / 2.0
        interpolated_points.append(interpolated_point)

    interpolated_points = np.concatenate(interpolated_points, axis=0)
    sampled_points = np.concatenate([points, interpolated_points], axis=0)

    return sampled_points


def filter_point_cloud(point_cloud, angle_threshold_degrees=90, fix=False):
    # Extract XYZ coordinates from the point cloud
    xyz = point_cloud[:, :3]

    # Center the point cloud at the origin and normalize
    center = np.mean(xyz, axis=0)
    normalized_xyz = (xyz - center) / np.linalg.norm(xyz - center, axis=1)[:, np.newaxis]

    # Randomly select a point within a sphere of radius
    if fix:
        angles = np.pi / 2
    else:
        angles = np.random.uniform(0, np.pi)
    x = np.sin(angles)
    y = np.cos(angles)
    random_point = np.array([x, y, 0])

    # Calculate angles between the random point and all points in the normalized point cloud
    angles = np.arccos(np.dot(normalized_xyz, random_point.T))

    # Filter points based on the angle threshold
    filtered_indices = np.where(np.degrees(angles) < angle_threshold_degrees)[0]

    # Keep only the selected points
    selected_points = point_cloud[filtered_indices]

    return selected_points


def occlusion(point_cloud, num_points=10000, angle_threshold_degrees=90, fix=False):
    filtered_point_cloud = filter_point_cloud(point_cloud, angle_threshold_degrees, fix)
    filtered_num = filtered_point_cloud.shape[0]
    k = int(num_points / filtered_num) + 1
    if 1 < k < 5:
        point_cloud = interpolate_point_cloud(filtered_point_cloud, k)
    return point_cloud


def rotation(xyz, rotation_angle):
    x, y, z = rotation_angle
    x, y, z = int(x), int(y), int(z)
    x_rad, y_rad, z_rad = np.radians(x), np.radians(y), np.radians(z)

    rot_x = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    rot_y = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    rot_z = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])

    rot_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
    xyz = np.matmul(xyz, rot_matrix)
    return xyz


def load_pts(path, separator=',', verbose=False):
    extension = path.split('.')[-1]
    if extension == 'npy':
        pcl = np.load(path, allow_pickle=True)
    elif extension == 'npz':
        pcl = np.load(path)
        pcl = pcl['pred']
    elif extension == 'ply':
        ply = PlyData.read(path)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pcl = np.column_stack((x, y, z))
        if len(vertex.properties) == 6:
            (r, g, b) = (vertex[t] for t in ('red', 'green', 'blue'))
            pcl = np.column_stack((pcl, r, g, b))
            pcl[:, 3:] = pcl[:, 3:] / 255 if pcl[:, 3:].max() > 1.0 else pcl
    elif extension == 'txt':
        f = open(path, 'r')
        line = f.readline()
        data = []
        while line:
            x, y, z = line.split(separator)[:3]
            data.append([float(x), float(y), float(z)])
            line = f.readline()
        f.close()
        pcl = np.array(data)
    elif extension == 'pth' or extension == 'pt':
        pcl = torch.load(path, map_location='cpu')
        pcl = pcl.detach().numpy()
    else:
        print('unsupported file format.')
        raise FileNotFoundError

    if len(pcl.shape) == 3:
        pcl = pcl[0]
        print("the dimension is 3, we select the first element in the batch.")

    if pcl.shape[0] == 3 or pcl.shape[0] == 6:
        pcl = pcl.T

    if verbose:
        print(f'point cloud shape: {pcl.shape}')
    assert pcl.shape[-1] == 3 or pcl.shape[-1] == 6

    return pcl


def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    return pc


def random_sample(pc, num):
    ori_num = pc.shape[0]
    if ori_num > num:
        permutation = np.arange(ori_num)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
    return pc


def process_pts(pts, data_args):
    pts = random_sample(pts, data_args.sample_points_num)
    pts[:, :3] = pc_norm(pts[:, :3])
    if data_args.with_color:
        if pts[:, 3:].max() > 1.0 + 1e-2:
            pts[:, 3:] = pts[:, 3:] / 255.0
    else:
        pts = pts[:, :3]
    pts = torch.from_numpy(pts).float()
    return pts


def tokenizer_point_token(prompt, tokenizer, point_token_index=POINT_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<point>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [point_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
