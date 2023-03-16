from configs.default import *
from configs.zjumocap.dymap_313 import *

exp_name = '392'
basedata_cfg = dict(
    data_root='data/my_zjumocap/my_392',
    human='my_392',
    ann_file='data/my_zjumocap/my_392/annots.npy',
    split='train'
)

train['scheduler'] = dict(
    type='warmup_exponential',
    gamma=0.1,
    decay_epochs=400
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})
