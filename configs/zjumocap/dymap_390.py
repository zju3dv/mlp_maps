from configs.default import *
from configs.zjumocap.dymap_313 import *

exp_name = '390'
basedata_cfg = dict(
    data_root='data/my_zjumocap/my_390',
    human='my_390',
    ann_file='data/my_zjumocap/my_390/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})
