import argparse
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, optax
from torch.utils.data import DataLoader
from optax.losses import softmax_cross_entropy_with_integer_labels as celoss

from untangle.utils import collect_information, function_error, best_of_n
from untangle.scaler import JacobianScaler
from untangle import algorithm

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

def collate(batch):
    samples, labels = zip(*batch)
    return jnp.stack(samples) / 255.0, jnp.stack(labels)

batch_size = 128

from untangle._common import default_dof

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, required=False, default=10)
    parser.add_argument('--ntries', type=int, required=False, default=5)
    args = parser.parse_args()

    niters = args.niters
    ntries = args.ntries

    key = jax.random.key(0)

    train, test = load_mnist()

    trainloader = DataLoader(train, batch_size=batch_size, collate_fn=collate)
    testloader  = DataLoader(test,  batch_size=batch_size, collate_fn=collate)

    params = jnp.load('./tmp/weights.npz')
    params = [(params['W1'], params['b1']), (params['W2'], params['b2'])]

    X = next(iter(trainloader))[0]
    N = X.shape[0]

    print(X.shape)
    print(default_dof(N))

    X, Y, J = collect_information(lambda x: forward(params, x), N, key, 784, X=X)

    scaler = JacobianScaler(J, Y)
    J_scaled, Y_scaled = scaler.scale()
    scaling_factors = scaler.factors

    def dforward(dparams, x):
        V, W = dparams
        return (W @ decoupling.internals(V.T @ x)) / scaling_factors

    for rank in [32]:

        print(f'RANK = {rank}')

        # compute decoupling

        min_err = jnp.inf
        best_decoupling = None

        for k in jax.random.split(key, ntries):
            decoupling, error = algorithm.cmtf_psd(X, Y_scaled, J_scaled, rank, niters, key=k)

            dparams = [decoupling.V, decoupling.W]

            errs = function_error(lambda x: forward(params, x), lambda x: dforward(dparams, x), X)
            print(jnp.mean(errs))

            if jnp.mean(errs) < min_err:
                best_decoupling = decoupling
                min_err = jnp.mean(errs)

        decoupling = best_decoupling
        (W, V, _, _) = decoupling.factors

        Z = X @ V
        O = jax.vmap(decoupling.internals)(Z)

        ncols = min(4, rank)
        nrows = (rank + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = axes.flatten()

        for r in range(rank):
            z = Z[:, r]
            y = O[:, r]
            idx = jnp.argsort(z)
            zs = z[idx]
            ys = y[idx]
            axes[r].plot(zs, ys, color='black', linewidth=3)
            axes[r].set_yticks([])
            axes[r].set_xticks([])

        plt.tight_layout()
        plt.savefig(f'internals{rank}.png', dpi=300)
        plt.close(fig)

        dparams = [V, W]

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
            print(f'\nAccuracy = {accuracy*100:.2f}% ===================\n')

        doptim = optax.adam(learning_rate=1e-3)
        dstate = doptim.init(dparams)

        @jax.jit
        def dstep(dparams, dstate, X, y):
            loss, grads = jax.value_and_grad(dcompute_loss)(dparams, X, y)
            updates, dstate = doptim.update(grads, dstate)
            dparams = optax.apply_updates(dparams, updates)
            return dparams, dstate, loss

        evaluate_decoupling(dparams, testloader)

        bar = tqdm(trainloader, desc='Fine-tuning')
        for _, (Xb, yb) in enumerate(bar):
            dparams, dstate, loss = dstep(dparams, dstate, Xb, yb)
            bar.set_postfix(loss=f'{loss:.4f}')

        evaluate_decoupling(dparams, testloader)

        preds_decoupling = jnp.argmax(batch_dforward(dparams, X), axis=-1)
        preds_nn = jnp.argmax(batch_predict(params, X), axis=-1)
        same = jnp.mean(preds_decoupling == preds_nn)
        print(f'Same = {same*100:.2f}%')

if __name__ == '__main__': main()

