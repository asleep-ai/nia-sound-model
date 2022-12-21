from torch import nn

class NeuralNetwork(nn.Module):
    """Define the neural network model."""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        model_size = 32
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40, model_size),
            nn.ReLU(),
            nn.Linear(model_size, 2)
        )

    def forward(self, x):
        """The forward pass of the model."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
