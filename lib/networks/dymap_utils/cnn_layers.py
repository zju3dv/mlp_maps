import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        out = x.view(self.shape)
        return out


class LinearELR(nn.Module):
    """Linear layer with equalized learning rate from stylegan2"""
    def __init__(self, inch, outch, lrmult=1., norm=None, act=None):
        super(LinearELR, self).__init__()

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        initgain = 1. / math.sqrt(inch)

        self.weight = nn.Parameter(torch.randn(outch, inch) / lrmult)
        self.weightgain = actgain

        if norm == None:
            self.weightgain = self.weightgain * initgain * lrmult

        self.bias = nn.Parameter(torch.full([outch], 0.))

        self.norm = norm
        self.act = act

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, norm={}, act={}'.format(
            self.weight.size(1), self.weight.size(0), self.norm, self.act
        )

    def getweight(self):
        if self.fused:
            return self.weight
        else:
            weight = self.weight
            if self.norm is not None:
                if self.norm == "demod":
                    weight = F.normalize(weight, dim=1)
            return weight

    def fuse(self):
        if not self.fused:
            with torch.no_grad():
                self.weight.data = self.getweight() * self.weightgain
        self.fused = True

    def forward(self, x):
        if self.fused:
            weight = self.getweight()

            out = torch.addmm(self.bias[None], x, weight.t())
            if self.act is not None:
                out = self.act(out)
            return out
        else:
            weight = self.getweight()

            if self.act is None:
                out = torch.addmm(self.bias[None], x, weight.t(), alpha=self.weightgain)
                return out
            else:
                out = F.linear(x, weight * self.weightgain, bias=self.bias)
                out = self.act(out)
                return out


def blockinit(k, stride):
    dim = k.ndim - 2
    return k \
            .view(k.size(0), k.size(1), *(x for i in range(dim) for x in (k.size(i+2), 1))) \
            .repeat(1, 1, *(x for i in range(dim) for x in (1, stride))) \
            .view(k.size(0), k.size(1), *(k.size(i+2)*stride for i in range(dim)))


class ConvTranspose2dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(ConvTranspose2dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 2 / (stride ** 2))

        initgain = stride if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(blockinit(
            torch.randn(inch, outch, kernel_size//self.stride, kernel_size//self.stride), self.stride))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [1, 3, 4]
                    else:
                        normdims = [0, 2, 3]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, :, None, None, None] # [B, inch, 1, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None:
            x = x.view(1, b * self.inch, x.size(2), x.size(3))
            weight = weight.view(b * self.inch, self.outch, self.kernel_size, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv_transpose2d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2), out.size(3))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None]
        else:
            bias = self.bias[None, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

class Conv2dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(Conv2dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 2)

        initgain = 1. if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(
            torch.randn(outch, inch, kernel_size, kernel_size))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [2, 3, 4]
                    else:
                        normdims = [1, 2, 3]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, None, :, None, None] # [B, 1, inch, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None:
            x = x.view(1, b * self.inch, x.size(2), x.size(3))
            weight = weight.view(b * self.outch, self.inch, self.kernel_size, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv2d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2), out.size(3))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None]
        else:
            bias = self.bias[None, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out
