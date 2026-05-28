## NN Compression

This example shows how tensor decoupling can be used for neural network compression.

- `train.py` trains the target network.
- `compress.py` computes the `CMTF-PSD` decoupling and fine-tunes the compressed model.

Additional required dependencies:
- optax (Adam)
- torch (DataLoader)
- torchvision (MNIST)
- matplotlib (plotting the internals)
