import torch
from torch import nn
import kilonerf_cuda

class FourierEmbedding(nn.Module):
    def __init__(self, num_input_channels, num_frequencies, log_sampling=True):
        super(FourierEmbedding, self).__init__()
        max_frequency = num_frequencies - 1
        if log_sampling:
            self.frequency_bands = 2. ** torch.linspace(0., max_frequency, steps=num_frequencies)
        else:
            self.frequency_bands = torch.linspace(0., max_frequency, steps=num_frequencies)
        self.num_frequencies = num_frequencies
        self.num_output_channels = (2 * num_frequencies + 1) * num_input_channels
    
    def forward(self, x, num_blocks=64, num_threads=256):
        sh = x.shape[:-1]
        self.frequency_bands = self.frequency_bands.to(x)
        x = kilonerf_cuda.compute_fourier_features(x.contiguous().view(-1), self.frequency_bands, num_blocks, num_threads, 'cuda')
        return x.view(*sh, self.num_output_channels)


class CudaMultiNetworkLinear(nn.Module):
    def __init__(self,
                 num_networks,
                 in_features,
                 out_features,
                 biases=False,
                 weights=True):
        super(CudaMultiNetworkLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        self.group_limits = [2048, 1024]  # tunable
        self.aux_index = kilonerf_cuda.init_multimatmul_magma_grouped(
            self.num_networks, self.out_features, self.in_features,
            self.group_limits)
        self.aux_index_backward = kilonerf_cuda.init_multimatmul_magma_grouped(
            self.num_networks, self.in_features, self.out_features,
            self.group_limits)
        if biases is False:
            self.biases = torch.zeros((num_networks, out_features))
        if weights is False:
            self.weights = torch.ones(
                (num_networks, in_features, out_features))

    def forward(self, x, weights, biases, batch_size_per_network):
        if weights is None:
            weights = self.weights.to(x)
        if biases is None:
            biases = self.biases.to(x)
        return AddMultiMatMul.apply(biases,\
                x.contiguous(), weights.contiguous(),\
                self.out_features, self.in_features,\
                batch_size_per_network,\
                self.group_limits, self.aux_index, self.aux_index_backward)


class AddMultiMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, biases, input_vectors, weights, out_features, in_features,
                batch_size_per_network, group_limits, aux_index,
                aux_index_backward):
        ctx.save_for_backward(biases, input_vectors, weights,
                              batch_size_per_network)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_limits = group_limits
        ctx.aux_index = aux_index
        ctx.aux_index_backward = aux_index_backward
        if biases is not None:
            return kilonerf_cuda.multimatmul_magma_grouped_static(
                biases, input_vectors, weights, out_features, in_features,
                batch_size_per_network, 4, 1024, group_limits, aux_index)
        else:
            return kilonerf_cuda.multimatmul_magma_grouped_static_without_bias(
                biases, input_vectors, weights, out_features, in_features,
                batch_size_per_network, 4, 1024, group_limits, aux_index)

    @staticmethod
    def backward(ctx, grad_output):
        biases, input_vectors, weights, batch_size_per_network = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_biases = None
        grad_input_vectors = None
        grad_weights = None

        grad_biases = kilonerf_cuda.multi_row_sum_reduction(
            grad_output, batch_size_per_network)

        grad_input_vectors = kilonerf_cuda.multimatmul_magma_grouped_static_without_bias_transposed_weights(
            biases, grad_output, weights, ctx.in_features, ctx.out_features,
            batch_size_per_network, 4, 1024, ctx.group_limits,
            ctx.aux_index_backward)

        grad_weights = kilonerf_cuda.multimatmul_A_transposed(
            input_vectors, grad_output, batch_size_per_network)

        return grad_biases, grad_input_vectors, grad_weights, None, None, None, None, None, None
