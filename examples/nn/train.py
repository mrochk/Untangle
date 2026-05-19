import argparse
from tqdm import tqdm
from pathlib import Path
from functools import partial
import jax, jax.numpy as jnp, optax
from torch.utils.data import DataLoader
from optax.losses import softmax_cross_entropy_with_integer_labels as celoss

tmpdir = Path('./tmp')

def lecun_init(m, n, key):
    scale = 1 / n 
    w_key, b_key = jax.random.split(key)
    weights = scale * jax.random.normal(w_key, (n, m))
    bias = scale * jax.random.normal(b_key, (n,))
    return weights, bias

def init_nn(sizes, key):
  keys = jax.random.split(key, len(sizes))
  return [lecun_init(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def forward(nn, x):
    for weights, bias in nn[:-1]:
        x = jax.nn.relu(weights @ x + bias)
    last_weights, last_bias = nn[-1]
    logits = last_weights @ x + last_bias
    return logits

def predict(nn, x): return jax.nn.softmax(forward(nn, x))

def batch_forward(nn, batch):
    return jax.vmap(partial(forward, nn))(batch)

def batch_predict(nn, batch):
    return jax.vmap(partial(predict, nn))(batch)

def load_mnist():
    from torchvision.datasets import MNIST
    from torchvision import transforms 

    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.flatten().numpy()])

    train = MNIST(root=tmpdir, train=True, download=True, transform=transform)
    test  = MNIST(root=tmpdir, train=False, download=True, transform=transform)
    return train, test

def compute_loss(nn, X, y):
    logits = batch_forward(nn, X)
    return jnp.mean(celoss(logits, y))

def collate(batch):
    samples, labels = zip(*batch)
    return jnp.stack(samples) / 255.0, jnp.stack(labels)

def evaluate(nn, loader):
    accuracy = 0.0
    for i, (X, y) in enumerate(tqdm(loader, desc='Evaluation')):
        preds = jnp.argmax(batch_predict(nn, X), -1)
        accuracy += (jnp.mean(preds == y) - accuracy) / (i+1)
    print(f'Accuracy = {accuracy*100:.2f}%')

def compare(f, nn, loader):
    same_all = 0.0
    for i, (X, y) in enumerate(tqdm(loader, desc='Evaluation')):
        preds_f = jnp.argmax(jax.nn.softmax(jax.vmap(f)(X)), -1)
        preds_nn = jnp.argmax(batch_predict(nn, X))
        same = (jnp.mean(preds_f == preds_nn))
        same_all += same
    print(f'Same = {same_all / (i + 1) * 100:.2f}%')

batch_size = 128
layer_sizes = [784, 512, 10]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, default=30)
    args = parser.parse_args()
    num_epochs = args.epochs

    key = jax.random.key(0)

    params = init_nn(layer_sizes, key)

    train, test = load_mnist()

    trainloader = DataLoader(train, batch_size=batch_size, collate_fn=collate)
    testloader  = DataLoader(test,  batch_size=batch_size, collate_fn=collate)

    @jax.jit
    def step(params, state, X, y):
        loss, grads = jax.value_and_grad(compute_loss)(params, X, y)
        updates, state = optim.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    optim = optax.adam(learning_rate=1e-3)
    state = optim.init(params)

    for epoch in range(num_epochs):
        bar = tqdm(trainloader, desc=f'Epoch {epoch+1} / {num_epochs}')
        for batch in bar:
            params, state, loss = step(params, state, *batch)
            bar.set_postfix_str(f'loss={loss:.2f}')
        evaluate(params, testloader)

    jnp.savez(tmpdir / 'weights.npz', 
              W1=params[0][0], 
              b1=params[0][1], 
              W2=params[1][0], 
              b2=params[1][1],
    )

if __name__ == '__main__': main()
