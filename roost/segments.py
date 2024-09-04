import torch
import torch.nn as nn
# from torch_scatter import scatter_add, scatter_max, scatter_mean


def scatter_mean(x, index, dim=0):
    out_size = index.max().item() + 1 
    sum_scatter = torch.zeros(out_size, dtype=x.dtype, device=x.device).scatter_add_(dim, index, x)
    ones = torch.ones_like(x, dtype=torch.float32)
    count_scatter = torch.zeros(out_size, dtype=torch.float32, device=x.device).scatter_add_(dim, index, ones)
    mean_scatter = sum_scatter / count_scatter.clamp(min=1)

    return mean_scatter
    
def scatter_add(x, index, dim=0):
    out_size = index.max().item() + 1  
    output = torch.zeros(out_size, dtype=x.dtype, device=x.device).scatter_add_(dim, index, x)

    return output

def scatter_max(x, index, dim=0):
    out_size = index.max().item() + 1  

    max_scatter = torch.full((out_size,), float('-inf'), dtype=x.dtype, device=x.device)

    for i in range(x.size(dim)):
        max_scatter[index[i]] = torch.max(max_scatter[index[i]], x[i])

    return max_scatter

 
class MeanPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, index):

        mean = scatter_mean(x, index, dim=0)

        return mean

    def __repr__(self):
        return self.__class__.__name__


class SumPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, index):

        mean = scatter_add(x, index, dim=0)

        return mean

    def __repr__(self):
        return self.__class__.__name__


class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x, index):
        """ forward pass """

        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x, index, weights):
        """ forward pass """

        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, activation=nn.LeakyReLU,
        batchnorm=False
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, activation=nn.ReLU, batchnorm=False):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns,
                                       self.res_fcs, self.acts):
            x = act(bn(fc(x)))+res_fc(x)

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__
