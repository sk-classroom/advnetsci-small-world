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

from assignment.assignment import (
    compute_small_world_coefficient,
    compute_global_clustering,
    compute_average_path_length,
)

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
random.seed(42)
g_er = igraph.Graph.Erdos_Renyi(n=n, p=p)

# Use the giant component for ER to ensure connectivity for path length
if not g_er.is_connected():
    g_er = g_er.connected_components().giant()

sigma_ring = compute_small_world_coefficient(g)
sigma_er = compute_small_world_coefficient(g_er)

C = compute_global_clustering(g)
L = compute_average_path_length(g)
avg_deg = 2 * g.ecount() / g.vcount()
C_rand = avg_deg / (n - 1)
L_rand = np.log(n) / np.log(avg_deg)

print(
    f"[Setup] Sigma — Ring: {sigma_ring:.3f}, ER: {sigma_er:.3f} "
    f"(n={n}, k={k}, p={p:.5f}; C={C:.3f}, L={L:.3f}, C_rand={C_rand:.5f}, L_rand={L_rand:.3f})"
)

# ------------------------------------------------------------
# Test 1 : Small-world coefficient for ring lattice should be >> 1
# ------------------------------------------------------------
print(f"[Test 1] Ring sigma >> 1: got {sigma_ring:.3f}")
assert sigma_ring > 3.0, f"Small-world coefficient for ring lattice should be greater than 3: {sigma_ring}"

# ------------------------------------------------------------
# Test 2 : Small-world coefficient for ER graph should be ~ 1
# ------------------------------------------------------------
print(f"[Test 2] ER sigma ~ 1: got {sigma_er:.3f}")
assert np.isclose(sigma_er, 1.0, atol=0.3), f"Small-world coefficient for ER graph should be close to 1: {sigma_er}"

# ------------------------------------------------------------
# Test 3 : Robustness on a disconnected graph (finite and positive sigma)
# ------------------------------------------------------------
g_disconnected = igraph.Graph.Ring(n, directed=False, mutual=False, circular=True)
g_disconnected.delete_edges([(0, 1), (n // 2, n // 2 - 1)])
sigma_disc = compute_small_world_coefficient(g_disconnected)
print(f"[Test 3] Disconnected graph sigma (should be finite): {sigma_disc}")
assert not np.isnan(sigma_disc) and not np.isinf(sigma_disc), "Sigma should be finite for a mildly disconnected graph"
assert sigma_disc >= 0.0, "Sigma should be non-negative"

# ------------------------------------------------------------
# Test 4 : Watts–Strogatz (small p) vs no rewiring (ring)
# ------------------------------------------------------------
p_ws = 0.05
random.seed(42)
g_ws = igraph.Graph.Watts_Strogatz(1, n, k // 2, p_ws)
if not g_ws.is_connected():
    g_ws = g_ws.connected_components().giant()

sigma_ws = compute_small_world_coefficient(g_ws)
print(f"[Test 4] WS sigma at p={p_ws}: {sigma_ws:.3f} vs Ring sigma {sigma_ring:.3f}")
assert sigma_ws > sigma_ring, f"Small-world coefficient for WS (p={p_ws}) should exceed the no-rewiring ring case: WS={sigma_ws}, Ring={sigma_ring}"
