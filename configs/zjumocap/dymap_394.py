from configs.default import *
from configs.zjumocap.dymap_313 import *

exp_name = '394'
basedata_cfg = dict(
    data_root='data/my_zjumocap/my_394',
    human='my_394',
    ann_file='data/my_zjumocap/my_394/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})
