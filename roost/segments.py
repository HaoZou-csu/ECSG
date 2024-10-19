import torch
import torch.nn as nn
# from torch_scatter import scatter_add, scatter_max, scatter_mean

try:
    # 尝试导入torch-scatter中的torch_max函数
    from torch_scatter import scatter_add, scatter_max, scatter_mean
    print("Successfully imported scatter_max from torch-scatter.")
    
except ImportError:

    def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand(other.size())
        return src
    
    
    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        index = broadcast(index, src, dim)
    
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)
    
    
    def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                     out= None,
                     dim_size=None) -> torch.Tensor:
    
    
        new_src = torch.empty_like(src)
    
        unique_indices = torch.unique(index)
        for idx in unique_indices:
            # Get the mask for the current index
            mask = index == idx
    
            # Find the maximum value in src for the current index
            max_value = src[mask].max()
    
            # Assign the maximum value to all positions in new_src corresponding to the current index
            new_src[mask] = max_value
    
        return new_src, None
    
    
    def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                     out= None,
                     dim_size=None) -> torch.Tensor:
        out = scatter_add(src, index, dim, out, dim_size)
        dim_size = out.size(dim)
    
        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1
    
        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        count = scatter_add(ones, index, index_dim, None, dim_size)
        count[count < 1] = 1
        count = broadcast(count, out, dim)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode='floor')
        return out


 
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

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
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

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
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
