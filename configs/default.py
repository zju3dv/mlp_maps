# task
task = 'nerf4d'

# mode
mode = 'train'

# experiment name
exp_name = 'hello'

# embedding
xyz_res = 10
view_res = 4
time_res = 10

# training
distributed = False
resume = True

filter_mask = False
erode_edge = True
mask_bkgd = True
body_sample_ratio = 0.5
face_sample_ratio = 0.

mask_loss = True
depth_loss = False
lpips_loss = False
distortion_loss = False

# evaluation
eval = False
skip_eval = False

# epoch
ep_iter = -1
save_ep = 100
save_latest_ep = 5
eval_ep = 1000

# trained model
trained_model_dir = 'data/trained_model'

# recorder
record_dir = 'data/record'
log_interval = 1
record_interval = 20

# result
result_dir = 'data/result'

fix_random = False

# rendering
N_rand = 1024
N_samples = 64
perturb = 1
white_bkgd = False
center_pixel = False

# novel view
fixed_time = False
fixed_view = False
render_views = 100

# visualization
store_rgb = True
store_depth = False
store_diff = False
store_acc = False
store_encoding = False
store_video = True
clean_img = True
vis_mode = 'test_view'
vis_training_view = False
vis_novel_view = False
vis_mesh = False
occ_grid = False


# weight loss
tv_weight = 0.1
mask_weight = 0.1
distortion_weight = 0.01
kldiv_weight = 1e-6
