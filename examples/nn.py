import argparse
from tqdm import tqdm
from functools import partial
import jax, jax.numpy as jnp, optax
from torch.utils.data import DataLoader
from optax.losses import softmax_cross_entropy_with_integer_labels as celoss

import matplotlib.pyplot as plt

from untangle.utils import collect_information, function_error
from untangle import algorithm

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

    train = MNIST(root='./tmp', train=True, download=True, transform=transform)
    test  = MNIST(root='./tmp', train=False, download=True, transform=transform)
    return train, test

def compute_loss(nn, X, y):
    logits = batch_forward(nn, X)
    return jnp.mean(celoss(logits, y))

batch_size = 128
layer_sizes = [784, 512, 10]

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

import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, default=15)
    parser.add_argument('--rank', type=int, required=False, default=32)
    parser.add_argument('--niters', type=int, required=False, default=10)
    args = parser.parse_args()
    num_epochs = args.epochs
    rank = args.rank
    niters = args.niters

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

    batches = []
    for batch in trainloader:
        batches.append(batch[0])
        if len(batches) == 1: break

    X = jnp.concatenate(batches)
    N = X.shape[0]

    print(X.shape)

    X, Y, J = collect_information(lambda x: forward(params, x), N, key, 784, X=X)

    decoupling = algorithm.cmtf_bsd(X, Y, J, rank=rank, niters=niters, key=key)

    error = function_error(lambda x: forward(params, x), decoupling, X)
    print(f'Decoupling error: {error}')

    (W, V, _, _) = decoupling.factors

    Z = X @ V
    Y = jax.vmap(decoupling.internals)(Z)
    fig, ax = plt.subplots(rank, figsize=(10, 2*rank))
    for r in range(rank):
        z = Z[:, r]
        y = Y[:, r]
        idx = jnp.argsort(z)
        zs = z[idx]
        ys = y[idx]
        #ax[r].plot(zs, ys, color=colors[r % (len(colors) - 1)], linewidth=3)
        ax[r].set_ylabel(f"$g{r+1}(z{r+1})$")
        ax[r].plot(zs, ys, color='black', linewidth=3)
        ax[r].set_yticks([])
        ax[r].set_xticks([])

    plt.tight_layout()
    plt.savefig('internals.png')
    plt.close(fig)

    dparams = [V, W]
    print(V.shape, W.shape)

    def dforward(dparams, x):
        V, W = dparams
        return W @ decoupling.internals(V.T @ x)

    print(jax.nn.softmax(dforward(dparams, X[0])), y[0])
    print(jax.nn.softmax(dforward(dparams, X[1])), y[1])
    print(jax.nn.softmax(dforward(dparams, X[2])), y[2])

    def batch_dforward(dparams, X):
        return jax.vmap(partial(dforward, dparams))(X)

    preds_decoupling = jnp.argmax(batch_dforward(dparams, X), axis=-1)
    preds_nn = jnp.argmax(batch_predict(params, X), axis=-1)
    same = jnp.mean(preds_decoupling == preds_nn)
    print(f'Same = {same*100:.2f}%')

    def dcompute_loss(dparams, X, y):
        logits = batch_dforward(dparams, X)
        return jnp.mean(celoss(logits, y))

    def evaluate_decoupling(dparams, loader):
        accuracy = 0.0
        for i, (X, y) in enumerate(tqdm(loader, desc='Decoupling Evaluation')):
            logits = batch_dforward(dparams, X)
            preds = jnp.argmax(jax.nn.softmax(logits), -1)
            accuracy += (jnp.mean(preds == y) - accuracy) / (i+1)
        print(f'Accuracy = {accuracy*100:.2f}%')

    evaluate_decoupling(dparams, testloader)

    doptim = optax.adam(learning_rate=1e-4)
    dstate = doptim.init(dparams)

    Wog = jnp.copy(W)
    Vog = jnp.copy(V)

    #@jax.jit
    def dstep(dparams, dstate, X, y):
        loss, grads = jax.value_and_grad(dcompute_loss)(dparams, X, y)
        updates, dstate = doptim.update(grads, dstate)
        dparams = optax.apply_updates(dparams, updates)
        return dparams, dstate, loss

    evaluate_decoupling(dparams, testloader)

    for i in range(10):

        bar = tqdm(trainloader, desc='Fine-tuning')
        for i, (X, y) in enumerate(bar):
            dparams, dstate, loss = dstep(dparams, dstate, X, y)
            bar.set_postfix(loss=f'{loss:.4f}')

            if ((i+1) % 31) == 0: evaluate_decoupling(dparams, testloader)

        evaluate_decoupling(dparams, testloader)

        preds_decoupling = jnp.argmax(batch_dforward(dparams, X), axis=-1)
        preds_nn = jnp.argmax(batch_predict(params, X), axis=-1)
        same = jnp.mean(preds_decoupling == preds_nn)
        print(f'Same = {same*100:.2f}%')

        V, W = dparams
        print(jnp.linalg.norm(W - Wog) / jnp.linalg.norm(W))
        print(jnp.linalg.norm(V - Vog) / jnp.linalg.norm(V))

if __name__ == '__main__': main()
