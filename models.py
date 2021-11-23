from torch import nn



class MLP(nn.Module):
    
    def __init__(self, input_dim=28*28, output_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.net(x)
        return y