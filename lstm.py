"""LSTM on MNIST — 3-layer LSTM for handwritten digit classification using PyTorch."""

import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
INPUT_DIM = 28        # Each row of the 28×28 image is one time-step feature
SEQ_DIM = 28          # 28 time steps (rows) per image
HIDDEN_DIM = 100
LAYER_DIM = 3         # 3 stacked LSTM layers
OUTPUT_DIM = 10       # Digits 0-9
BATCH_SIZE = 100
N_ITERS = 3000
LEARNING_RATE = 0.1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """Stacked LSTM followed by a fully-connected classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialise hidden & cell states with zeros on the same device as x
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Use the hidden state from the last time step
        out = self.fc(out[:, -1, :])
        return out


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device):
    """Run the model on the test set and return accuracy (%)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, SEQ_DIM, INPUT_DIM).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    """Train the model and print test accuracy every 500 iterations."""
    iter_count = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, SEQ_DIM, INPUT_DIM).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter_count += 1

            if iter_count % 500 == 0:
                accuracy = evaluate(model, test_loader, device)
                print(
                    f"Iteration: {iter_count}. "
                    f"Loss: {loss.item():.4f}. "
                    f"Accuracy: {accuracy:.2f}%"
                )
                # Resume training mode after evaluation
                model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve data directory relative to this script
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Step 1 — Load MNIST dataset
    train_dataset = dsets.MNIST(
        root=data_dir, train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root=data_dir, train=False, transform=transforms.ToTensor()
    )

    num_epochs = int(N_ITERS / (len(train_dataset) / BATCH_SIZE))

    # Step 2 — Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Step 3 — Build model, loss, and optimizer
    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, LAYER_DIM, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Step 4 — Train
    train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)


if __name__ == "__main__":
    main()
