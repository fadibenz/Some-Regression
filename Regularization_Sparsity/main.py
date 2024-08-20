import matplotlib.pyplot as plt
import numpy as np

def calculate_norm(W, p):
    return np.sum(np.abs(W) ** p) ** (1/p)


X = np.linspace(-2, 2, 100)
Y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(X, Y)
pos = np.array([X, Y]).T
pos = pos.reshape(-1, 2)


normed_ws_half = np.array([calculate_norm(w, 0.5) for w in pos])
normed_ws_1 = np.array([calculate_norm(w, 1) for w in pos])
normed_ws_2 = np.array([calculate_norm(w, 2) for w in pos])

normed_ws_half = normed_ws_half.reshape(X.shape[0], X.shape[1])
normed_ws_1 = normed_ws_1.reshape(X.shape[0], X.shape[1])
normed_ws_2 = normed_ws_2.reshape(X.shape[0], X.shape[1])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot norm 0.5
ax[0].contourf(X, Y, normed_ws_half, cmap='viridis')
ax[0].set_title('Norm p=0.5')
ax[0].grid(True)

# Plot norm 1
ax[1].contourf(X, Y, normed_ws_1, cmap='viridis')
ax[1].set_title('Norm p=1')
ax[1].grid(True)

# Plot norm 2
ax[2].contourf(X, Y, normed_ws_2, cmap='viridis')
ax[2].set_title('Norm p=2')
ax[2].grid(True)


plt.tight_layout()
plt.show()
