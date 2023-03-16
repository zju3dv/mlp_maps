from configs.default import *
from configs.nhr.sport1 import *

exp_name = 'basketball'

def get_mask(data_root, img_path):
    msk_path = img_path_to_schp_path(img_path)
    msk_path = os.path.join(data_root, msk_path)
    msk = imageio.imread(msk_path)[..., :3].astype(np.int32)
    msk = (msk * [-1, 10, 100]).sum(axis=-1)

    palette = np.array([0, 128, 1280, 1408, 12800, 12928, 14080, 14208, 64,
                        192, 1344, 1472, 12864, 12992, 14144, 14272, 640, 768,
                        1920, 2048])
    leg_msk = (msk == palette[9]) | (msk == palette[16]) | (msk == palette[18])

    msk_path = img_path.replace('images', 'mask')
    msk_path = os.path.join(data_root, msk_path)
    msk = imageio.imread(msk_path)[..., 0].astype(np.int32)
    msk = (msk > 100).astype(np.uint8)
    msk[leg_msk] = 1

    return msk

basedata_cfg = dict(
    data_root='data/NHR/basketball_easymocap',
    human='basketball',
    ann_file='data/NHR/basketball_easymocap/annots.npy',
    split='train'
)

train_dataset = basedata_cfg.copy()
test_dataset = basedata_cfg.copy()
test_dataset.update({'split': 'test'})

# render options
render_path = [23, 22, 15, 14, 13, 12, 5, 4, 63, 62, 55, 54, 47, 46, 45, 44, 65, 64, 39, 38, 37, 36, 29, 28]
max_samples = 288
points_buffer = 400000

# interactive options
interactive_scale = 0.8
interactive_shift = np.array([0., 0., 0.3])

# data options
transform = np.array([[ 0.90630779, -0.07338689,  0.41619774,  2.],
               [ 0.42261826,  0.1573787 , -0.89253894, -5.13],
               [ 0.        ,  0.98480775,  0.17364818,  2.85],
               [ 0.        ,  0.        ,  0.        ,  1.]])


remove_view = [0, 9, 18]
test_view = [x for x in test_view if x not in remove_view]
