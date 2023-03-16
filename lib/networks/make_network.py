import os
import imp


def make_network(cfg):
    path = cfg.network_path
    module = path[:-3].replace('/', '.')
    network = imp.load_source(module, path).Network()
    return network
