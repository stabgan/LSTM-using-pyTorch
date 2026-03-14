# 🧠 LSTM on MNIST — PyTorch

A 3-layer LSTM network for handwritten digit classification on the MNIST dataset, built with PyTorch.

The model treats each 28×28 image as a sequence of 28 time steps with 28 features each, feeding it through a stacked LSTM to classify digits 0–9.

## Architecture

| Component | Detail |
|-----------|--------|
| Input | 28×28 MNIST images (28 time steps × 28 features) |
| LSTM | 3 layers, 100 hidden units, `batch_first=True` |
| Output | Fully connected → 10 classes |
| Optimizer | SGD (lr=0.1) |
| Loss | CrossEntropyLoss |
| Epochs | 5 (~3000 iterations) |

## Tech Stack

| | Technology |
|---|---|
| 🐍 | Python 3.x |
| 🔥 | PyTorch |
| 🖼️ | torchvision |
| 🧮 | CUDA (optional, auto-detected) |

## Requirements

```bash
pip install torch torchvision
```

## Usage

```bash
python lstm.py
```

MNIST data downloads automatically to `./data/` on first run.

The model trains for 5 epochs and prints test accuracy every 500 iterations.

## Results

Reaches ~98% accuracy on the MNIST test set after training.

## Known Issues

- Training runs on CPU by default if CUDA is not available — this is slow but functional.
- No learning rate scheduler; accuracy may plateau with longer training.
- No model checkpointing — training restarts from scratch each run.

## License

MIT — see [LICENSE](LICENSE)

## Author

**Kaustabh Ganguly** ([@stabgan](https://github.com/stabgan))
