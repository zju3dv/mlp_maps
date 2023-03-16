import os
import imp


def make_visualizer(cfg):
    path = cfg.visualizer_path
    module = path[:-3].replace('/', '.')
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer
