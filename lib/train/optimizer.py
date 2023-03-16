import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}

def make_params(net, lr, weight_decay):
    params = []
    if isinstance(lr, dict):
        for k in lr.keys():
            if not k.startswith('lr_'):
                continue

            prefix = k[len('lr_'):]
            params_k = [v for k, v in net.named_parameters() if k.startswith(prefix)]
            if len(params_k) == 0:
                continue
            
            lr_ = getattr(lr, f'lr_{prefix}')
            if lr_ > 0:
                for param_k in params_k:
                    params.append({'params': param_k, 'lr': lr_, 'weight_decay': weight_decay})
            else:
                for param_k in params_k:
                    param_k.requires_grad = False
    else:
        for key, value in net.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    return params

def make_optimizer(cfg, net, lr=None, weight_decay=None):
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    params = make_params(net, lr, weight_decay)
    if 'adam' in cfg.train.optim:
        if isinstance(lr, dict):
            optimizer = _optimizer_factory[cfg.train.optim](params, 5e-4, weight_decay=weight_decay)
        else:
            optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
