import imp

def make_renderer(cfg, network):
    path = cfg.renderer_path
    module = path[:-3].replace('/', '.')
    renderer = imp.load_source(module, path).Renderer(network)
    return renderer
