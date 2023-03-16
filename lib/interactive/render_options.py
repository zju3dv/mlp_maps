# this is the render option class, just a dot dict
# it should control all modifiable render options through the imgui options

from lib.utils.base_utils import DotDict
from lib.config import cfg


opt = DotDict()

# -----------------------------------------------------------------------------
# * Interactive Rendering Related
# -----------------------------------------------------------------------------
opt.fps_cnter_int = 1  # update fps per 0.5 seconds
opt.render_level = 1  # indexing rendering scale
opt.type = 0  # indexing rendering scale
opt.type_mapping = ['pred', 'depth', 'seg', 'bbox']

h, w = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
opt.window_hw = [h, w]

opt.font_filepath = 'lib/interactive/fonts/Caskaydia Cove Nerd Font Complete.ttf'
# opt.lock_fxfy = True
opt.autoplay = True

opt.smoothing_term = 0.1
