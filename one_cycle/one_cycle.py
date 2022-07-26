import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset


class NeuralNetwork(nn.Module):
    """Define the neural network model.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        model_size = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, model_size),
            nn.ReLU(),
            nn.Linear(model_size, model_size),
            nn.ReLU(),
            nn.Linear(model_size, 2)
        )

    def forward(self, x):
        """The forward pass of the model.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)

        # forward pass
        pred = model(X)

        # compute loss
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred, y, pred.argmax(1), pred.argmax(1) == y)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"[Test ] Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


if __name__ == '__main__':
    batch_size = 8

    # Create the dataset for training and testing
    training_data = dataset.SleepSoundDataset(root="data", train=True)
    test_data = dataset.SleepSoundDataset(root="data", train=False)

    # Create a DataLoader for the training and test data
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get resources to run the experiment.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create the model
    model = NeuralNetwork().to(device)

    # Create the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        if epoch % 200 == 0:
            print(f"Epoch {epoch} -------------------------------")
            print(f"[Train] Loss: {train_loss:>7f}")
            test(test_dataloader, model, loss_fn)
    print("Done!")
