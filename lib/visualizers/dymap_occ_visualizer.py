import torch
from lib.config import cfg
import os
from lib.utils.net_utils import startswith_any
from termcolor import colored

torch.nn.Module.load_state_dict

class Visualizer:
    def __init__(self):
        if cfg.test.epoch == -1:
            self.model_path = os.path.join(cfg.trained_model_dir, 'latest.pth')
        else:
            self.model_path = os.path.join(cfg.trained_model_dir, '{}.pth'.format(cfg.test.epoch))
        assert os.path.exists(self.model_path), '{} doese not exist'.format(self.model_path)
        
        self.result_path = os.path.join(cfg.trained_model_dir, 'final.pth')
        print(colored('the results are saved at {}'.format(self.result_path), 'yellow'))
        self.occupancy_grids = []
        self.encodings = []

    def visualize(self, output, batch):
        occupancy_grid = output['occupancy_grid']
        encoding = output['encoding']
        self.occupancy_grids.append(occupancy_grid)
        self.encodings.append(encoding)

    def summarize(self):
        occupancy_grids = torch.stack(self.occupancy_grids, dim=0).bool()
        encodings = torch.stack(self.encodings, dim=0)
        pretrained_model = torch.load(self.model_path)
        net = pretrained_model['net']
        removed_keys = ['latent_vector']
        for k in list(net.keys()):
            if startswith_any(k, removed_keys):
                del net[k]
        
        meta = {
            'occupancy_grids': occupancy_grids,
            'encodings': encodings,
        }
        net.update(meta)
        final_model = {'net': net, 'epoch': 0}
        
        torch.save(final_model, self.result_path)
        
        
                
