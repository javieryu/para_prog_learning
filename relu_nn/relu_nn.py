from torch import nn


class FFReLUNet(nn.Module):
    """
    Implements a basic feed forward neural network that uses
    ReLU activations for all of the layers.
    """
    def __init__(self, shape):
        """ Constructor for network.

        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(FFReLUNet, self).__init__()
        self.shape = shape

        # Build up the layers
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(nn.Linear(shape[i], shape[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass on the input through the network.

        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]

        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        return self.layers[-1](x)
