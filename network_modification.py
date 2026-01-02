import random

import numpy as np
from matplotlib import pyplot as plt

import data_management
import data_management as dm

PLOT = True

"""
The algorithm should work in the follwing way: 
inputs: 
amount of nodes (total 246) 
Structural connectome [normalised 0-1] (246x246 np matrix) 
node atlas (246*3) x, y, z 
posistion of nodes 
n_path_steps 
function: 
-for each brain node 
-use the connectome get the amount of non zero elements (ex a row is for one node) 
-use the connectome and get the paths (a path is a array with the nodes) that 
end after n_path_steps and has connections to them (exlude paths that include the same node again) 
-use the node atlas to calculate all paths length to the final node in the path from prevoius step 
-sort the calculated path lengs by the shortest first 
-reconect the connections of the current nodes to the shortest path length ones, 
keep the same amount of connections and values as the original connectome 
-return the new connectome and a touple list 
with [(i,j), (n,m)] where (i,j) is original connection and (n,m) is reconnected connectiion.
"""

def extract_coord():
    BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)
    coordinates_arr = []
    for row in BNA_atlas:
        coordinates_arr.append([row[2], row[3], row[4]])
    return np.array(coordinates_arr)

def path_distance(path, coords):
    dist = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dist += np.linalg.norm(coords[b] - coords[a])
    return dist


def find_paths(start, C, n_steps):
    paths = []
    def dfs(node, depth, path):
        if depth == n_steps:
            paths.append(path.copy())
            return
        for nbr in np.where(C[node] > 0)[0]:
            if nbr not in path:
                dfs(nbr, depth + 1, path + [nbr])
    dfs(start, 0, [start])
    return paths


def rewire_connectome_symmetric(C, coords, n_path_steps):
    N = C.shape[0]
    # -------------------------------------------------------------------------
    #  STEP 1 — each node proposes new neighbors
    # -------------------------------------------------------------------------
    connections = np.sum(C > 0, axis=1)
    unity = np.ones((N, N))

    new_C = C.copy()
    rewired = []
    index = [j for j in range(N)]
    random.shuffle(index)

    for i in range(N):
        current_paths = find_paths(index[i], C, n_path_steps)
        potential_paths = find_paths(index[i], unity, n_path_steps)
        end_terminals = [p[n_path_steps] for p in current_paths]

        scored = []
        for p in potential_paths:
            if p[-1] in end_terminals:
                dist = path_distance(p, coords)
                scored.append((dist, p))

        scored.sort(key=lambda x: x[0])

        suggested_connections = [np[1][2] for np in scored[:connections[index[i]]]]
        current_connections = [cp[1] for cp in current_paths]

        remove = []
        for cc in current_connections:
            if cc not in suggested_connections:
                remove.append(cc)


        for sp in suggested_connections:
            if sp not in current_connections:
                new_C[index[i]][sp] = C[index[i]][remove[0]]
                new_C[sp][index[i]] = C[index[i]][remove[0]]
                new_C[index[i]][remove[0]] = 0
                new_C[remove[0]][index[i]] = 0
                rewired.append([(index[i], remove[0]), (index[i], sp)])
                remove.pop(0)

    return new_C, rewired



###From SOCS lecutre 12
def path_length(A, i, j):
    """
    Function returning the minimum path length between thwo nodes.

    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    i, j : Nodes indices.
    """

    Lij = - 1

    if A[i, j] > 0:
        Lij = 1
    else:
        N = np.size(A[0, :])
        P = np.zeros([N, N]) + A
        n = 1
        running = True
        while running:
            P = np.matmul(P, A)
            n += 1
            running
            if P[i, j] > 0:
                Lij = n
            if (n > N) or (Lij > 0):
                running = False

    return Lij

def matrix_path_length(A):
    """
    Function returning a matrix L of minimum path length between nodes.

    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """

    N = np.size(A[0, :])
    L = np.zeros([N, N]) - 1

    for i in range(N):
        for j in range(i + 1, N):
            L[i, j] = path_length(A, i, j)
            L[j, i] = L[i, j]

    return L


def nodes_degree(A):
    """
    Function returning the degree of a node.

    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """

    degree = np.sum(A, axis=0)

    return degree

def clustering_coefficient(A):
    """
    Function returning the clustering coefficient of a graph.

    Parameters
    ==========
    A : Adjacency matrix (assumed symmetric).
    """

    K = nodes_degree(A)
    N = np.size(K)

    C_n = np.sum(np.diagonal(np.linalg.matrix_power(A, 3)))
    C_d = np.sum(K * (K - 1))

    C = C_n / C_d

    return C

def show(file=0, file_numbers = 100):
        connections =np.fromfile("data/Modified_DTI/connections.bin", dtype=float).reshape((2,100))
        cc = np.fromfile("data/Modified_DTI/cc.bin", dtype=float).reshape((2,100))
        total_degree=np.fromfile("data/Modified_DTI/total_degree.bin", dtype=float).reshape((2,100))
        avg_degree=np.fromfile("data/Modified_DTI/avg_degree.bin", dtype=float).reshape((2,100))
        diameter=np.fromfile("data/Modified_DTI/diameter.bin", dtype=float).reshape((2,100))
        mean_path_length=np.fromfile("data/Modified_DTI/mean_path_length.bin", dtype=float).reshape((2,100))

        print(f"connections:{connections[0,:].sum()/file_numbers} connections rewired:{connections[1,:].sum()/file_numbers}")
        print(f"clustering_coefficient:{cc[0,:].sum()/file_numbers} clustering_coefficient rewired: {cc[1,:].sum()/file_numbers}")
        print(f"total degree: {total_degree[0,:].sum()/file_numbers} total degree rewired: {total_degree[1,:].sum()/file_numbers}")
        print(f"avg degree: {avg_degree[0,:].sum()/file_numbers} avg degree rewired: {avg_degree[1,:].sum()/file_numbers}")
        print(f"Diameter:{diameter[0,:].sum()/file_numbers} Diameter rewired: {diameter[1,:].sum()/file_numbers}")
        print(f"Mean path length:{mean_path_length[0,:].sum()/file_numbers} Mean path length rewired: {mean_path_length[1,:].sum()/file_numbers}")


        if PLOT:
            individuals = [ind for ind in range(file_numbers)]

            fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            # ---- Compute means ----
            mean_cc_actual = np.mean(cc[0, :file_numbers])
            mean_cc_spi = np.mean(cc[1, :file_numbers])

            mean_mpl_actual = np.mean(mean_path_length[0, :file_numbers])
            mean_mpl_spi = np.mean(mean_path_length[1, :file_numbers])


            # ---- Row 1: Clustering Coefficient (CC) ----
            axs[0].plot(individuals, cc[0, :file_numbers],
                        label="Actual Clustering Coefficient")
            axs[0].plot(individuals, cc[1, :file_numbers],
                        label="SPI Clustering Coefficient")
            axs[0].set_title("Clustering Coefficient (CC)")
            axs[0].grid()
            axs[0].legend()

            # Add mean text
            axs[0].text(
                0.98, 0.95,
                f"Mean Actual: {mean_cc_actual:.4f}\nMean SPI: {mean_cc_spi:.4f}",
                transform=axs[0].transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # ---- Row 2: Mean Path Length (MPL) ----
            axs[1].plot(individuals, mean_path_length[0, :file_numbers],
                        label="Actual Mean Path Length")
            axs[1].plot(individuals, mean_path_length[1, :file_numbers],
                        label="SPI Mean Path Length")
            axs[1].set_title("Mean Path Length (MPL)")
            axs[1].grid()
            axs[1].legend()

            # Add mean text
            axs[1].text(
                0.98, 0.95,
                f"Mean Actual: {mean_mpl_actual:.4f}\nMean SPI: {mean_mpl_spi:.4f}",
                transform=axs[1].transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # ---- Shared labels ----
            axs[1].set_xlabel(r"$I[n]$ - individuals")
            fig.supylabel(r"CC and MPL Values for $n = 10$ individual")
            fig.suptitle("CC and MPL of Actual and Shortest Path Ideal (SPI) Connectome")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("Figures/100_individuals_change.png")

            N = 246

            new_C = np.fromfile(f"data/Modified_DTI/mDTI_individual_{file}.bin").reshape((N, N))
            C = dm.load_DTI_data(file)

            # Circular layout
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

            x = np.cos(theta)
            y = np.sin(theta)

            plt.figure(figsize=(8, 8))

            M = C > 0
            M_rewired = new_C > 0

            # Draw nodes
            plt.scatter(x, y, s=10, c='blue', zorder=3)
            # for i in range(N):
            # plt.text(x[i] * 1.05, y[i] * 1.05, str(i), ha='center', va='center')

            # Draw edges
            for i in range(N):
                for j in range(i + 1, N):  # only upper triangle to avoid duplicates
                    if M[i, j] and M_rewired[i, j]:
                        # Edge unchanged
                        plt.plot([x[i], x[j]], [y[i], y[j]], 'k-', linewidth=0.5, zorder=1)
                    elif M[i, j] and not M_rewired[i, j]:
                        # Edge changed (removed)
                        plt.plot([x[i], x[j]], [y[i], y[j]], 'r-', linewidth=0.5, zorder=1)
                    elif not M[i, j] and M_rewired[i, j]:
                        # Edge changed (added)
                        plt.plot([x[i], x[j]], [y[i], y[j]], 'g-', linewidth=0.5, zorder=1)

            plt.axis('off')
            plt.title("Unchanged (black), removed (red) and added (green) Edges")
            plt.savefig("Figures/modification_to_SC_t.png", transparent=True)
            plt.savefig("Figures/modification_to_SC.png")
            #plt.show()

def calc(files):

    connections = np.zeros((2, 100))
    cc = np.zeros((2, 100))
    total_degree = np.zeros((2, 100))
    avg_degree = np.zeros((2, 100))
    diameter = np.zeros((2, 100))
    mean_path_length = np.zeros((2, 100))

    coordinates = extract_coord()
    for i in range(files):
        print(i)
        C = data_management.load_DTI_data(i)

        # =====================================================
        # Create a 10-node sample connectome
        # =====================================================
        """
        np.random.seed(42)
        N = 246

        coords = np.random.rand(N, 3)

        C = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if np.random.rand() < 0.05:
                    w = np.random.rand()
                    C[i, j] = C[j, i] = w
        """
        coords = coordinates

        # =====================================================
        # Apply rewiring
        # =====================================================
        new_C, rewired = rewire_connectome_symmetric(C, coords, n_path_steps=2)

        new_C.tofile(f"data/Modified_DTI/mDTI_individual_{i}.bin")

        A = matrix_path_length(C)
        A_rewired = matrix_path_length(new_C)

        connections[0, i] = np.sum(new_C > 0)
        connections[1, i] = np.sum(C > 0)

        cc[0, i] = clustering_coefficient(C)
        cc[1, i] = clustering_coefficient(new_C)

        total_degree[0, i] = np.sum(nodes_degree(C))
        total_degree[1, i] = np.sum(nodes_degree(new_C))

        avg_degree[0, i] = np.mean(nodes_degree(C))
        avg_degree[1, i] = np.mean(nodes_degree(new_C))

        diameter[0, i] = np.max(A)
        diameter[1, i] = np.max(A_rewired)

        mean_path_length[0, i] = np.mean(A[A > 0])
        mean_path_length[1, i] = np.mean(A_rewired[A_rewired > 0])

    connections.tofile("data/Modified_DTI/connections.bin")
    cc.tofile("data/Modified_DTI/cc.bin")
    total_degree.tofile("data/Modified_DTI/total_degree.bin")
    avg_degree.tofile("data/Modified_DTI/avg_degree.bin")
    diameter.tofile("data/Modified_DTI/diameter.bin")
    mean_path_length.tofile("data/Modified_DTI/mean_path_length.bin")

if __name__ == "__main__":
    #calc(100)
    show(0, file_numbers=100)




