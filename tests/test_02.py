# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.2.6",
#     "python-igraph==0.11.9",
# ]
# ///

# %% Import
import numpy as np
import sys
import os
import igraph
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assignment.assignment_teacher import compute_average_path_length

# %% Test ------------
# Create initial ring lattice (p=0)
n = 300
k = 4
g = igraph.Graph.Ring(n, directed=False, mutual=False, circular=True)

# Add additional edges to make each node have degree k
for i in range(n):
    for j in range(2, k // 2 + 1):
        neighbor = (i + j) % n
        if not g.are_adjacent(i, neighbor):
            g.add_edge(i, neighbor)

p = g.density()
# Set the random seed for reproducibility
random.seed(42)
g_er = igraph.Graph.Erdos_Renyi(n=n, p=p)

# Use the giant component to ensure connectivity for path length
if not g_er.is_connected():
    g_er = g_er.connected_components().giant()

L = compute_average_path_length(g)
L_er = compute_average_path_length(g_er)

print(f"[Setup] Computed average path lengths — Ring: {L:.3f}, ER: {L_er:.3f} (n={n}, k={k}, p={p:.5f})")

# ------------------------------------------------------------
# Test 1 : Does the function handle disconnected graphs?
# ------------------------------------------------------------
# Does the function handle disconnected graphs?
g_disconnected = igraph.Graph.Ring(n, directed=False, mutual=False, circular=True)
g_disconnected.delete_edges([(0, 1), (n //2, n//2  - 1)])
L_disconnected = compute_average_path_length(g_disconnected)
print("[Test 1] Disconnected graph handling: average path length =", L_disconnected)

assert ~np.isinf(L_disconnected), f"Average path length is infinite. Hint: It is likely that the average path length is infinite because the graph is not connected. Your implementation should be robust to the disconnected case by removing the path lengths for node pairs that are not connected."

# ------------------------------------------------------------
# Test 2 : Does the function excluded the distance to the node itself?
# ------------------------------------------------------------
# Does the function excluded the distance to the node itself?
g_n4 = igraph.Graph.Ring(5, directed=False, mutual=False, circular=True)
L_n4 = compute_average_path_length(g_n4)
print("[Test 2] Self-distance exclusion: L_n4 =", L_n4)

assert np.isclose(L_n4, 1.5), f"Average path length of a ring lattice with 5 nodes should be 1.5: {L_n4}. Hint: If the length is less than 1.5, it is likely that the function does not exclude the distance to the node itself."


# ------------------------------------------------------------
# Test 3 : Does the function return the correct average path length for the Erdős–Rényi random graph?
# ------------------------------------------------------------

L_er_exp = np.log(n) / np.log(k)
print(f"[Test 3] ER average path length: expected {L_er_exp:.3f}, got {L_er:.3f}")
assert np.abs(L_er_exp - L_er) < 0.5, f"Average path length of the Erdős–Rényi random graph should be close to {L_er_exp}: {L_er}"

# ------------------------------------------------------------
# Test 4 : Does the average path length of the Ring lattice is close to the expected value?
# ------------------------------------------------------------
L_ring_exp = 37.876254180602004
print(f"[Test 4] Ring lattice average path length: expected {L_ring_exp:.3f}, got {L:.3f}")
assert np.isclose(L, L_ring_exp), f"Average path length of the ring lattice should be close to {L_ring_exp}: {L}"
