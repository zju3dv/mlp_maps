import imp

def _evaluator_factory(cfg):
    path = cfg.evaluator_path
    module = path[:-3].replace('/', '.')
    evaluator = imp.load_source(module, path).Evaluator()
    return evaluator

def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
