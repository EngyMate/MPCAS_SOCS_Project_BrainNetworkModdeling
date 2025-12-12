import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

import data_management
import data_management as dm


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

    new_C = C.copy()
    rewired = []

    for i in range(N):
        current_paths = find_paths(i, C, n_path_steps)
        potential_paths = find_paths(i, np.ones((N,N)), n_path_steps)
        end_terminals = [p[n_path_steps] for p in current_paths]

        scored = []
        for p in potential_paths:
            dist = path_distance(p, coords)
            scored.append((dist, p))

        scored.sort(key=lambda x: x[0])

        suggested_connections = [np[1][2] for np in scored[:connections[i]]]
        current_connections = [cp[1] for cp in current_paths]

        remove = []
        for cc in current_connections:
            if cc not in suggested_connections:
                remove.append(cc)


        for sp in suggested_connections:
            if sp not in current_connections:
                new_C[i][sp] = C[i][remove[0]]
                new_C[sp][i] = C[i][remove[0]]
                new_C[i][remove[0]] = 0
                new_C[remove[0]][i] = 0
                rewired.append([(i, remove[0]), (i, sp)])
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

if __name__ == "__main__":
    BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)

    coordinates = extract_coord()

    dti = data_management.load_DTI_data(0)

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
    C = dti
    coords = coordinates
    N = C.shape[0]
    # =====================================================
    # Apply rewiring
    # =====================================================
    new_C, rewired = rewire_connectome_symmetric(C, coords, n_path_steps=2)
    print(f"Rewired:{len(rewired)}/{np.sum(C > 0)}")
    print(f"Current:{np.sum(new_C > 0)}")
    """
    for r in rewired:
        print(r)
    print("\n")
    """

    print(f"clustering_coefficient:{clustering_coefficient(C)} clustering_coefficient rewired: {clustering_coefficient(new_C)}")
    print(f"total degree: {np.sum(nodes_degree(C))} total degree rewired: {np.sum(nodes_degree(new_C))}")
    print(f"avg degree: {np.mean(nodes_degree(C))} avg degree rewired: {np.mean(nodes_degree(new_C))}")
    A = matrix_path_length(C)
    A_rewired = matrix_path_length(new_C)
    print(f"Diameter:{np.max(A)} Diameter rewired: {np.max(A_rewired)}")
    print(f"Mean path length:{np.mean(A[A > 0])} Mean path length rewired: {np.mean(A_rewired[A_rewired > 0])}")


    #new_C_2, rewired_2 = rewire_connectome_symmetric(new_C, coords, n_path_steps=2)

    # Circular layout
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(8, 8))

    M = C > 0
    M_rewired = new_C > 0

    # Draw nodes
    plt.scatter(x, y, s=10, c='lightblue', zorder=3)
    for i in range(N):
        plt.text(x[i] * 1.05, y[i] * 1.05, str(i), ha='center', va='center')

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):  # only upper triangle to avoid duplicates
            if M[i, j] and M_rewired[i, j]:
                # Edge unchanged
                plt.plot([x[i], x[j]], [y[i], y[j]], 'k-', linewidth=0.2, zorder=1)
            elif  M[i, j] and not M_rewired[i, j]:
                # Edge changed (removed)
                plt.plot([x[i], x[j]], [y[i], y[j]], 'r-', linewidth=0.2, zorder=1)
            elif not M[i, j] and M_rewired[i, j]:
                # Edge changed (added)
                plt.plot([x[i], x[j]], [y[i], y[j]], 'g-', linewidth=0.2, zorder=1)

    plt.axis('off')
    plt.title("Unchanged (black), removed (red) and added (green) Edges")
    plt.show()


