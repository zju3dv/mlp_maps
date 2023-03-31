from lib.config import cfg
import os
from termcolor import colored

class Visualizer:
    def __init__(self):
        self.result_dir = os.path.join(cfg.result_dir, 'mesh')
        print(colored('the results are saved at {}'.format(self.result_dir), 'yellow'))
        os.system('mkdir -p {}'.format(self.result_dir))
        
    def visualize(self, output, batch):
        mesh = output['mesh']
        i = batch['frame_index'].item()
        mesh_path = os.path.join(self.result_dir, '{:04d}.ply'.format(i))
        mesh.export(mesh_path)
