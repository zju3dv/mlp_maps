from configs.default import *
from configs.nhr.sport1 import *

exp_name = 'sport2'

basedata_cfg = dict(
    data_root='data/NHR/sport_2_easymocap',
    human='sport_2',
    ann_file='data/NHR/sport_2_easymocap/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})

transform = np.array([[ 0.90630779, -0.07338689,  0.41619774,  2.],
               [ 0.42261826,  0.1573787 , -0.89253894, -5.13],
               [ 0.        ,  0.98480775,  0.17364818,  2.85],
               [ 0.        ,  0.        ,  0.        ,  1.]])