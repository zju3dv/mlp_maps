import os
from configs.default import *

task = 'zjumocap'
gpus = [0]

exp_name = '313'

# functions
img_path_to_smpl_ind = lambda x: int(os.path.basename(x)[:-4])
img_path_to_frame_ind = lambda x: int(os.path.basename(x)[:-4])
img_path_to_schp_path = lambda x: 'schp' + x[6:-4] + '.png'
img_path_to_mask_path = lambda x: 'mask' + x[6:-4] + '.png'


train_dataset_path = 'lib/datasets/dymap.py'
test_dataset_path = 'lib/datasets/dymap.py'

network_path = 'lib/networks/dymap.py'
renderer_path = 'lib/networks/renderer/dymap.py'
trainer_path = 'lib/train/trainers/dymap.py'

evaluator_path = 'lib/evaluators/if_nerf.py'
visualizer_path = 'lib/visualizers/dymap.py'

basedata_cfg = dict(
    data_root='data/my_zjumocap/my_313',
    human='my_313',
    ann_file='data/my_zjumocap/my_313/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})

train = dict(
    batch_size=16,
    collator='',
    optim='adam',
    lr={
        'lr_latent_vectors': 5e-4,
        'lr_plane_decoder': 5e-4,
        'lr_hashtable': 5e-3,
    },
    weight_decay=0,
    epoch=400,
    scheduler=dict(
        type='warmup_exponential',
        gamma=0.1,
        decay_epochs=400
    ),
    batch_sampler='default',
    num_workers=8,
    shuffle=True,
)

test = dict(
    sampler='',
    frame_sampler_interval=30,
    batch_size=1,
    collator='',
    batch_sampler='default',
    epoch=-1,
)


ep_iter = 500
save_ep = 100
eval_ep = 1000

# network options
use_encoder = False
fixedcameras = [0, 4, 8]
code_dim = 256

mlp_use_bias = False
mlp_map_size = 16
mlp_overlap = True

mlp_layers = 3
layer_size = 32

plane_size = 256

hash = {
    'level': 16,
    'size': 19,
    'dim': 2,
}

feature_channel = 32
assert feature_channel == hash['level'] * hash['dim']
feature_plane_types = ['xy', 'xz', 'yz']
feature_num_planes = len(feature_plane_types)

kn_plane_types = ['xy', 'xz', 'yz']
kn_num_planes = len(kn_plane_types)

view_res = 2

# render options
fast_render = False
max_samples = 256
weight_thres = 1e-3
ERT = False
adjust_hsv = False
points_buffer = 200000

# data options
H = 1024
W = 1024
ratio = 0.5
N_rand = 1024
face_sample_ratio = 0.1
body_sample_ratio = 0.7
training_view = list(range(0, 21, 2))
test_view = [x for x in range(21) if x not in training_view]

num_train_frame = 300
num_eval_frame = 1
begin_ith_frame = 0
frame_interval = 1

vertices = 'vertices'
big_box = True
box_padding = 0.1

# training options
mask_weight = 0.1
kldiv_weight = 1e-6


# novel view cfg
novel_view_cfg = dict(
    test_dataset_path='lib/datasets/dymap_novel_dataset.py',
    vis_mode='novel_view'
)


grid_cfg = dict(
    test_dataset_path='lib/datasets/dymap_occ_dataset.py',
    renderer_path='lib/networks/renderer/dymap_occ_renderer.py',
    visualizer_path='lib/visualizers/dymap_occ_visualizer.py',
    grid_resolution = [24, 24, 48],
    subgrid_resolution=[5, 5, 5],
    alpha_thres = 5,
)

mesh_cfg = dict(
    test_dataset_path='lib/datasets/dymap_occ_dataset.py',
    renderer_path='lib/networks/renderer/mesh_renderer.py',
    visualizer_path='lib/visualizers/mesh_visualizer.py',
    voxel_size = [0.005, 0.005, 0.005],
    mesh_th = 5,
    mesh_begin_ith_frame = 0,
    mesh_num_train_frame = 10,
    mesh_frame_interval = 30,
)