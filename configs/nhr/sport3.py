from configs.default import *
from configs.nhr.sport1 import *

exp_name = 'sport3'

basedata_cfg = dict(
    data_root='data/NHR/sport_3_easymocap',
    human='sport_3',
    ann_file='data/NHR/sport_3_easymocap/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})

transform = np.array([[ 0.98480775, -0.04494346,  0.16773126, -0.75454014],
       [-0.17364818, -0.254887  ,  0.95125124, -4.9550868 ],
       [ 0.        , -0.96592583, -0.25881905,  1.9544816 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
