import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_spiral(n_points=200, noise=0.1, seed=42):
    """Genere deux spirales entrelacees : classe 0 et classe 1."""
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

X, y = generate_spiral(n_points=400, noise=0.15)

# Architecture 2-64-64-1, initialisation He
np.random.seed(42)
W1 = np.random.randn(2, 64) * np.sqrt(2 / 2);   b1 = np.zeros(64)
W2 = np.random.randn(64, 64) * np.sqrt(2 / 64); b2 = np.zeros(64)
W3 = np.random.randn(64, 1) * np.sqrt(2 / 64);  b3 = np.zeros(1)

lr = 0.1         
n_epochs = 10000
losses = []

for epoch in range(n_epochs):
    # Forward
    z1 = np.dot(X, W1) + b1;  a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2; a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3; y_pred = sigmoid(z3).flatten()

    loss = bce_loss(y, y_pred)
    losses.append(loss)

    # Backward
    n = len(y)
    err3 = (y_pred - y).reshape(-1, 1)             # sortie
    dW3 = (1 / n) * np.dot(a2.T, err3); db3 = np.mean(err3, axis=0)

    err2 = np.dot(err3, W3.T) * relu_grad(z2)      # couche 2
    dW2 = (1 / n) * np.dot(a1.T, err2); db2 = np.mean(err2, axis=0)

    err1 = np.dot(err2, W2.T) * relu_grad(z1)      # couche 1
    dW1 = (1 / n) * np.dot(X.T, err1); db1 = np.mean(err1, axis=0)

    # Update
    W3 -= lr * dW3; b3 -= lr * db3
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 500 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# Frontiere de decision
h = 0.02
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
                     np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h))
grid = np.c_[xx.ravel(), yy.ravel()]
a1g = relu(np.dot(grid, W1) + b1)
a2g = relu(np.dot(a1g, W2) + b2)
zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=10, edgecolors='none')
axes[0].set_title("Frontiere de decision (2-64-64-1)")
axes[1].plot(losses)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss spirale")
plt.savefig("phase4_spirale.png", dpi=100, bbox_inches='tight')

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y):.2%}")
