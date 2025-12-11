def watts_strogatz_sw(n, c, p):
    import numpy as np

    A = np.zeros([n, n])
    P = np.random.rand(n, n)
    A_rewired = np.zeros([n, n])
    rewired_edges = np.zeros([n, n])

    c_half = c // 2

    # Regular ring lattice
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            A[i, j % n] = 1
            A[j % n, i] = 1

    # Rewire
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            original = j % n

            if P[i, original] < p:
                k = (np.random.randint(n - 1) + 1 + i) % n
                A_rewired[i, k] = 1
                A_rewired[k, i] = 1
                rewired_edges[i, k] = 1
                rewired_edges[k, i] = 1
            else:
                A_rewired[i, original] = 1
                A_rewired[original, i] = 1

    # Circular layout
    x = np.cos(np.arange(n) / n * 2 * np.pi)
    y = np.sin(np.arange(n) / n * 2 * np.pi)

    return A_rewired, rewired_edges, x, y

import matplotlib.pyplot as plt
import numpy as np

n = 246
c = 4

p1 = 0.6   # rewiring for BLACK edges (baseline)
p2 = 0.1   # rewiring for RED edges (highlighted)

# ---- First pass: black network ----
A1, rewired1, x, y = watts_strogatz_sw(n, c, p1)

plt.figure(figsize=(8, 8))

# Draw ALL edges in black
for i in range(n):
    for j in range(i+1, n):
        if A1[i, j] == 1:
            plt.plot([x[i], x[j]], [y[i], y[j]], '-', linewidth=1)

# ---- Second pass: red rewired edges only ----
A2, rewired2, _, _ = watts_strogatz_sw(n, c, p2)

# Draw only rewired edges in red
for i in range(n):
    for j in range(i+1, n):
        if rewired2[i, j] == 1:       # only new rewired edges
            plt.plot([x[i], x[j]], [y[i], y[j]], 'r-', linewidth=2)

# Draw nodes
plt.plot(x, y, 'o', color='blue')

#plt.title("Structural connectome network\nBlack = empirical, Red = rewired")
plt.axis('off')
plt.savefig("Plots/Poster/connectome_graph", transparent=True)
