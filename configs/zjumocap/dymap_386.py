from configs.default import *
from configs.zjumocap.dymap_313 import *

exp_name = '386'
basedata_cfg = dict(
    data_root='data/my_zjumocap/my_386',
    human='my_386',
    ann_file='data/my_zjumocap/my_386/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})
