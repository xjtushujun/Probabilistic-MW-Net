import torch.nn as nn


class VNet(nn.Module):
    def __init__(self, n_hidden):
        super(VNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            # nn.ReLU(),
            # nn.Sigmoid(),
            nn.Tanhshrink(),
            nn.Linear(n_hidden, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.mlp(x)


class VNet2(nn.Module):
    def __init__(self, n_hidden):
        super(VNet2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanhshrink(),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(n_hidden, 2),
            nn.Softplus()
        )

    def forward(self, x):
        return self.mlp(x)


class VNet3(nn.Module):
    def __init__(self, n_hidden):
        super(VNet3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanhshrink(),
            # nn.PReLU(),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(n_hidden, 2),
            nn.Softplus()
        )

    def forward(self, x):
        return self.mlp(x)
