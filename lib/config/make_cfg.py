from .config import Config
import argparse
import os
from types import ModuleType
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.py', type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = Config.fromfile(args.config)
cfg.merge_from_list(args.opts)

os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
cfg.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else args.local_rank

cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)

if cfg.get('fast_render', False):
    cfg.renderer_path = 'lib/networks/renderer/dymap_fast_renderer.py'
if cfg.get('vis_training_view', False):
    cfg.vis_mode = 'training_view'
if cfg.get('vis_novel_view', False):
    cfg.update(cfg.novel_view_cfg)
if cfg.get('vis_mesh', False):
    cfg.update(cfg.mesh_cfg)
if cfg.get('occ_grid', False):
    cfg.update(cfg.grid_cfg)



dumped_cfg = cfg.to_dict()
dumped_cfg = {k: v for k, v in dumped_cfg.items() if not isinstance(v, ModuleType)}
dumped_cfg = {k: v for k, v in dumped_cfg.items() if not callable(v)}
os.system('mkdir -p {}'.format(cfg.trained_model_dir))
np.save(os.path.join(cfg.trained_model_dir, 'cfg.npy'), dumped_cfg)
