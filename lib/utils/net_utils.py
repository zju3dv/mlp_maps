import torch
import os
import torch.nn.functional
from lib.config import cfg
from collections import OrderedDict
from termcolor import colored
import time

def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        # os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth not in ['latest.pth', 'final.pth'] and '.pth' in pth
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth not in ['latest.pth', 'final.pth'] and '.pth' in pth
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def startswith_any(k, l):
    for s in l:
        if k.startswith(s):
            return True
    return False


def load_network(net, model_dir, resume=True, epoch=-1, strict=True, only=[]):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        if cfg.get('fast_render', False):
            model_path = os.path.join(model_dir, 'final.pth')
        else:            
            pths = [
                int(pth.split('.')[0]) for pth in os.listdir(model_dir)
                if pth not in ['latest.pth', 'final.pth'] and '.pth' in pth
            ]
            if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
                return 0
            if epoch == -1:
                if 'latest.pth' in os.listdir(model_dir):
                    pth = 'latest'
                else:
                    pth = max(pths)
            else:
                pth = epoch
            model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    pretrained_net = pretrained_model['net']

    if only:
        strict = False
        keys = list(pretrained_net.keys())
        for k in keys:
            if not startswith_any(k, only):
                del pretrained_net[k]
                
    net.load_state_dict(pretrained_net, strict=strict)

    return pretrained_model['epoch'] + 1


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


def requires_grad(m, req):
    for param in m.parameters():
        param.requires_grad = req


class perf_timer:
    def __init__(self, msg="Elapsed time: {}s", logf=lambda x: print(colored(x, 'yellow')), sync_cuda=True, use_ms=False, disabled=False):
        self.logf = logf
        self.msg = msg
        self.sync_cuda = sync_cuda
        self.use_ms = use_ms
        self.disabled = disabled

        self.loggedtime = None

    def __enter__(self,):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.logtime(self.msg)

    def logtime(self, msg=None, logf=None):
        if self.disabled:
            return
        # SAME CLASS, DIFFERENT FUNCTIONALITY, is this good?
        # call the logger for timing code sections
        if self.sync_cuda:
            torch.cuda.synchronize()

        # always remember current time
        prev = self.loggedtime
        self.loggedtime = time.perf_counter()

        # print it if we've remembered previous time
        if prev is not None and msg:
            logf = logf or self.logf
            diff = self.loggedtime-prev
            diff *= 1000 if self.use_ms else 1
            logf(msg.format(diff))

        return self.loggedtime
