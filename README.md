# 🧠 LSTM on MNIST — PyTorch

A 3-layer stacked LSTM for handwritten digit classification on MNIST, built with PyTorch.

Each 28×28 image is treated as a sequence of 28 time steps with 28 features per step. The stacked LSTM processes this sequence and a fully-connected layer maps the final hidden state to one of 10 digit classes.

## Architecture

| Component | Detail |
|-----------|--------|
| Input | 28×28 MNIST images (28 time steps × 28 features) |
| LSTM | 3 layers, 100 hidden units, `batch_first=True` |
| Output | Fully connected → 10 classes |
| Optimizer | SGD (lr = 0.1) |
| Loss | CrossEntropyLoss |
| Epochs | ~5 (3 000 iterations) |

## 🛠 Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🐍 | Python 3.8+ | Runtime |
| 🔥 | PyTorch | Deep learning framework |
| 🖼️ | torchvision | MNIST dataset & transforms |
| 🧮 | CUDA | Optional GPU acceleration |

## Requirements

```bash
pip install torch torchvision
```

## Usage

```bash
python lstm.py
```

MNIST downloads automatically to a `data/` directory next to the script on first run. The model trains for ~5 epochs and prints test accuracy every 500 iterations.

## Results

Reaches **~98 %** accuracy on the MNIST test set.

## ⚠️ Known Issues

- Training is slow on CPU — a CUDA GPU is recommended.
- No learning-rate scheduler; accuracy may plateau with longer training.
- No model checkpointing — training restarts from scratch each run.

## License

MIT — see [LICENSE](LICENSE)

## Author

**Kaustabh Ganguly** ([@stabgan](https://github.com/stabgan))
