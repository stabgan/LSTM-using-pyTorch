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

## Requirements

- Python 3.x
- PyTorch
- torchvision

```bash
pip install torch torchvision
```

## Usage

```bash
python lstm.py
```

MNIST data will be downloaded automatically to `./data/` on first run.

The model trains for 5 epochs and prints test accuracy every 500 iterations.

## Results

The model reaches **~98% accuracy** on the MNIST test set after training.

![results](https://image.ibb.co/gMpuv7/Screen_Shot_2018_03_09_at_2_16_31_PM.png)

## Known Issues

- `loss.data[0]` is deprecated in PyTorch ≥ 0.5 — replace with `loss.item()`
- `Variable` wrapper is deprecated since PyTorch 0.4 — tensors have autograd built-in now
- No `model.eval()` or `torch.no_grad()` during test evaluation

## License

MIT — see [LICENSE](LICENSE)

## Author

**Kaustabh Ganguly** ([@stabgan](https://github.com/stabgan))
