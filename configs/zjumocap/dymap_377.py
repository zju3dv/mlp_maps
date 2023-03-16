from configs.default import *
from configs.zjumocap.dymap_313 import *

exp_name = '377'
basedata_cfg = dict(
    data_root='data/my_zjumocap/my_377',
    human='my_377',
    ann_file='data/my_zjumocap/my_377/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})
