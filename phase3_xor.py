import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_xor(seed, learning_rate=1.0, n_epochs=10000):
    """Entraine un reseau 2-2-1. Retourne poids, losses, accuracy finale."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2, 2))     # init plus large que 0.5 -> aide a demarrer
    b1 = np.zeros(2)
    W2 = rng.standard_normal((2, 1))
    b2 = np.zeros(1)
    losses = []
    for epoch in range(n_epochs):
        # Forward
        a1 = sigmoid(np.dot(X_xor, W1) + b1)      # couche cachee [4, 2]
        a2 = sigmoid(np.dot(a1, W2) + b2)         # sortie [4, 1]
        y_pred = a2.flatten()
        losses.append(compute_loss_bce(y_xor, y_pred))
y_xor = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_xor(seed, learning_rate=1.0, n_epochs=10000):
    """Entraine un reseau 2-2-1. Retourne poids, losses, accuracy finale."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2, 2))     # init plus large que 0.5 -> aide a demarrer
    b1 = np.zeros(2)
    W2 = rng.standard_normal((2, 1))
    b2 = np.zeros(1)
    losses = []
    for epoch in range(n_epochs):
        # Forward
        a1 = sigmoid(np.dot(X_xor, W1) + b1)      # couche cachee [4, 2]
        a2 = sigmoid(np.dot(a1, W2) + b2)         # sortie [4, 1]
        y_pred = a2.flatten()
        losses.append(compute_loss_bce(y_xor, y_pred))
        # Backprop couche 2
        error2 = (y_pred - y_xor).reshape(-1, 1)
        dW2 = (1 / 4) * np.dot(a1.T, error2)
        db2 = np.mean(error2, axis=0)
        # Backprop couche 1 (chain rule)
        error1 = np.dot(error2, W2.T) * (a1 * (1 - a1))
        dW1 = (1 / 4) * np.dot(X_xor.T, error1)
        db1 = np.mean(error1, axis=0)
        # Update
        W2 -= learning_rate * dW2; b2 -= learning_rate * db2
        W1 -= learning_rate * dW1; b1 -= learning_rate * db1
    acc = np.mean((y_pred > 0.5) == y_xor)
    return (W1, b1, W2, b2), losses, acc

# Redemarrages : on essaie plusieurs graines, on garde la premiere qui atteint 100%
for seed in range(30):
    (W1, b1, W2, b2), losses, acc = train_xor(seed)
    print(f"seed {seed:2d} | Loss finale: {losses[-1]:.4f} | Accuracy: {acc:.2%}")
    if acc == 1.0:
        print(f"\n>>> Convergence a 100% avec seed={seed}")
        break

# Frontiere de decision
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, cmap='RdBu', edgecolors='k')
plt.title("XOR : frontiere de decision du reseau 2-2-1")
plt.savefig("phase3_xor_boundary.png", dpi=100, bbox_inches='tight')

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {acc:.2%}")
