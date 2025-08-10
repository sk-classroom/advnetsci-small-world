# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.1",
#     "python-igraph==0.11.9",
#     "pyarrow",
#     "openai==1.99.5",
# ]
# ///

import marimo
import numpy as np

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Small-World Networks Assignment

    Welcome to the Small-World Networks assignment! You'll learn about the fascinating properties of small-world networks through hands-on implementation and interactive visualization.

    Small-world networks exhibit two key properties:

    - **High clustering**: Nodes tend to form triangles (friends of friends are friends)
    - **Short path lengths**: Despite high clustering, nodes are connected by short paths

    This assignment will guide you through implementing key network metrics and exploring the Watts-Strogatz model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Understanding Small-World Networks

    Small-world networks were first studied by **Watts and Strogatz (1998)**. These networks interpolate between regular lattices and random graphs, exhibiting:

    1. **High clustering coefficient** like regular networks
    2. **Short average path length** like random networks

    This combination creates the "small-world" phenomenon where most nodes can be reached through a small number of steps, despite high local clustering.

    ---

    # 📋 Assignment Instructions & Grading
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📋 Assignment Tasks": mo.md(
                r"""
            Complete the following tasks and upload your notebook to your GitHub repository.

            1. **Task 1**: Implement `compute_global_clustering(g)` - Calculate the global clustering coefficient
            2. **Task 2**: Implement `compute_average_path_length(g)` - Calculate the average shortest path length
            3. **Task 3**: Implement `compute_small_world_coefficient(g)` - Calculate the small-world coefficient using random graph as reference
            4. Update this notebook by using `git add`, `git commit`, and then `git push`.
            5. The notebook will be automatically graded, and your score will be shown on GitHub.
            """
            ),
            "🔒 Protected Files": mo.md(
                r"""
            Protected files are test files and configuration files you cannot modify. They appear in your repository but don't make any changes to them.
            """
            ),
            "⚖️ Academic Integrity": mo.md(
                r"""
            There is a system that automatically checks code similarity across all submissions and online repositories. Sharing code or submitting copied work will result in zero credit and disciplinary action.

            While you can discuss concepts, each student must write their own code. Cite any external resources, including AI tools, in your comments.
            """
            ),
            "📚 Allowed Libraries": mo.md(
                r"""
            You **cannot** import any other libraries that result in the grading script failing or a zero score. Only use: `numpy`, `igraph`, `altair`, `pandas`.
            """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 1: Global Clustering Coefficient

    The **global clustering coefficient** measures the overall tendency of nodes to cluster together. It's defined as:

    **C_global = (3 × number of triangles) / (number of connected triples)**

    Where:
    - A **triangle** is a set of 3 fully connected nodes
    - A **connected triple** is a set of 3 nodes where at least 2 edges exist

    Research how to implement this calculation efficiently.
    """
    )
    return


@app.function
# Task 1
def compute_global_clustering(g):
    """
    Compute the global clustering coefficient of a graph.

    Args:
        g (igraph.Graph): Input graph

    Returns:
        float: Global clustering coefficient (0.0 to 1.0)
    """
    return g.transitivity_undirected()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 2: Average Path Length

    The **average path length** (also called **characteristic path length**) is the average number of edges in the shortest paths between all pairs of nodes.
    """
    )
    return


@app.function
# Task 2
def compute_average_path_length(g):
    """
    Compute the average shortest path length of a graph.

    Args:
        g (igraph.Graph): Input graph (should be connected)

    Returns:
        float: Average path length
    """
    distances = g.distances()
    n = g.vcount()

    distances = np.array(distances)
    triu_indices = np.triu_indices(n, k=1)
    distances = distances[triu_indices]
    distances = distances[~np.isin(distances, np.inf)]
    return np.mean(distances)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 3: Small-World Coefficient

    The **small-world coefficient** (σ) quantifies how "small-world" a network is by comparing it to equivalent random and regular networks:

    $$
    σ = \dfrac{(C/C_\text{random})}{(L/L_\text{random})}
    $$

    Where:

    - C = clustering coefficient of the network
    - C_random = clustering coefficient of equivalent random network
    - L = average path length of the network
    - L_random = average path length of equivalent random network

    **$\sigma \gg 1$** indicates small-world properties (high clustering but short paths like random graphs).

    For the random reference, use the Erdős–Rényi random graph model. See the lecture notes for the formula.
    """
    )
    return


@app.function
# Task 3
def compute_small_world_coefficient(g):
    """
    Compute the small-world coefficient using random graph as reference.

    Args:
        g (igraph.Graph): Input graph

    Returns:
        float: Small-world coefficient (σ)
    """
    # Get basic properties
    n = g.vcount()
    m = g.ecount()

    # Compute C and L for input graph
    C = compute_global_clustering(g)
    L = compute_average_path_length(g)

    # Compute C_random and L_random
    average_deg = 2 * m / n
    C_random = average_deg / (n - 1)
    L_random = np.log(n) / np.log(average_deg)

    # Avoid division by zero
    if C_random == 0 or L_random == 0:
        return 0.0

    # Return small-world coefficient
    return (C / C_random) / (L / L_random)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Interactive Visualization: Watts-Strogatz Model

    Now let's explore how these metrics change as we vary the rewiring probability in the Watts-Strogatz model!

    The **Watts-Strogatz model** starts with a ring lattice and randomly rewires edges with probability `p`:
    - `p = 0`: Regular ring lattice (high clustering, long paths)
    - `p = 1`: Random graph (low clustering, short paths)
    - `0 < p < 1`: Small-world region (high clustering AND short paths)

    Use the slider below to explore different rewiring probabilities:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Rewiring probability slider
    p_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.1, label="Rewiring Probability (p)"
    )
    p_slider
    return (p_slider,)


@app.cell(hide_code=True)
def _(network_chart):
    network_chart
    return


@app.cell(hide_code=True)
def _(clustering_chart, mo, path_chart, sigma_chart):
    mo.hstack(
        [sigma_chart, mo.vstack([clustering_chart, path_chart])],
        justify="center",
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Understanding the Visualization

    The interactive plots above show key insights about small-world networks:

    1. **Network Structure**: Shows how edges get rewired from regular (gray) to random (red) connections
    2. **Clustering Coefficient**: Blue line showing how local clustering changes with rewiring
    3. **Average Path Length**: Red line showing how global connectivity changes with rewiring
    4. **Small-World Coefficient**: Orange line showing the small-world property peak

    ### Key Observations:
    - At **p = 0**: High clustering, long paths (regular lattice)
    - At **p = 1**: Low clustering, short paths (random graph)
    - At **intermediate p**: High clustering AND short paths (small-world!)

    The small-world regime occurs when clustering decreases slowly but path length drops rapidly as p increases.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Libraries""")
    return


@app.cell
def _():
    # All imports in one place to avoid conflicts
    import numpy as np
    import igraph
    import altair as alt
    import pandas as pd
    return alt, igraph, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Generate Watts-Strogatz Networks""")
    return


@app.cell(hide_code=True)
def _(compute_small_world_coefficient, igraph, np, pd):
    # Pre-compute all statistics using progressive rewiring
    def generate_ws_metrics_progressive(p_values, n=300, k=4, seed=42):
        """Generate metrics by progressively rewiring a single ring lattice"""
        np.random.seed(seed)
        results = []

        # Create initial ring lattice (p=0)
        g = igraph.Graph.Ring(n, directed=False, mutual=False, circular=True)

        # Add additional edges to make each node have degree k
        for i in range(n):
            for j in range(2, k // 2 + 1):
                neighbor = (i + j) % n
                if not g.are_adjacent(i, neighbor):
                    g.add_edge(i, neighbor)

        # Get initial edges and prepare for progressive rewiring
        original_edges = [(e.source, e.target) for e in g.es]
        total_edges = len(original_edges)

        # Pre-determine complete rewiring order
        edges_to_rewire_order = np.random.choice(
            total_edges, size=total_edges, replace=False
        )
        edges_rewired_so_far = 0

        for p in p_values:
            target_rewires = int(p * total_edges)

            # Rewire additional edges to reach target_rewires
            while edges_rewired_so_far < target_rewires:
                edge_idx = edges_to_rewire_order[edges_rewired_so_far]
                u, v = original_edges[edge_idx]

                # Find the edge in current graph
                edge_id = None
                for e in g.es:
                    if (e.source == u and e.target == v) or (
                        e.source == v and e.target == u
                    ):
                        edge_id = e.index
                        break

                if edge_id is not None:  # Edge still exists
                    # Remove the edge
                    g.delete_edges([edge_id])

                    # Find valid rewiring targets
                    possible_targets = []
                    for node in range(n):
                        if node != u and not g.are_adjacent(u, node):
                            possible_targets.append(node)

                    if possible_targets:
                        new_target = np.random.choice(possible_targets)
                        g.add_edge(u, new_target)
                    else:
                        # If no valid target, restore original edge
                        g.add_edge(u, v)

                edges_rewired_so_far += 1

            # Ensure graph is still connected for meaningful metrics
            if g.is_connected():
                try:
                    # Compute metrics using the implemented functions
                    C = compute_global_clustering(g)
                    L = compute_average_path_length(g)
                    sigma = compute_small_world_coefficient(g)

                    results.append(
                        {
                            "p": p,
                            "clustering": C,
                            "path_length": L,
                            "small_world": sigma,
                            "edges_rewired": edges_rewired_so_far,
                        }
                    )
                except:
                    # Skip if computation fails
                    continue

        return pd.DataFrame(results)


    # Generate all data once using progressive rewiring
    p_values = np.linspace(0, 1, 50)  # From 0 to 1.0
    ws_data_full = generate_ws_metrics_progressive(p_values)
    return (ws_data_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Filter Data""")
    return


@app.cell(hide_code=True)
def _(p_slider, pd, ws_data_full):
    # Filter data up to current slider value
    current_p = p_slider.value
    ws_data = (
        ws_data_full[ws_data_full["p"] <= current_p].copy()
        if len(ws_data_full) > 0
        else pd.DataFrame()
    )

    # Find the closest point in pre-computed data for current marker
    if len(ws_data_full) > 0:
        closest_idx = (ws_data_full["p"] > current_p).idxmax() - 1
        current_data = ws_data_full.iloc[[closest_idx]].copy()
    else:
        current_data = pd.DataFrame()
    return current_data, current_p, ws_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create Network Visualization""")
    return


@app.cell(hide_code=True)
def _(alt, current_p, igraph, np, pd):
    # Create network visualization
    def create_network_at_p(p, n=300, k=4, seed=42):
        """Create a network at specific p value using progressive rewiring"""
        np.random.seed(seed)

        # Create initial ring lattice (p=0)
        g = igraph.Graph.Ring(n, directed=False, mutual=False, circular=True)

        # Add additional edges to make each node have degree k
        for i in range(n):
            for j in range(2, k // 2 + 1):
                neighbor = (i + j) % n
                if not g.are_adjacent(i, neighbor):
                    g.add_edge(i, neighbor)

        # Get initial edges and prepare for progressive rewiring
        original_edges = [(e.source, e.target) for e in g.es]
        total_edges = len(original_edges)

        # Rewire edges up to target p
        target_rewires = int(p * total_edges)
        edges_to_rewire_order = np.random.choice(
            total_edges, size=total_edges, replace=False
        )

        for i in range(target_rewires):
            edge_idx = edges_to_rewire_order[i]
            u, v = original_edges[edge_idx]

            # Find the edge in current graph
            edge_id = None
            for e in g.es:
                if (e.source == u and e.target == v) or (
                    e.source == v and e.target == u
                ):
                    edge_id = e.index
                    break

            if edge_id is not None:
                g.delete_edges([edge_id])

                # Find valid rewiring targets
                possible_targets = []
                for node in range(n):
                    if node != u and not g.are_adjacent(u, node):
                        possible_targets.append(node)

                if possible_targets:
                    new_target = np.random.choice(possible_targets)
                    g.add_edge(u, new_target)
                else:
                    g.add_edge(u, v)  # Restore if no valid target

        return g


    def create_network_data(g, n, k):
        """Create network visualization data"""
        # Create circular layout
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

        # Create nodes dataframe
        nodes_data = []
        for node in g.vs:
            x, y = pos[node.index]
            nodes_data.append({"node": node.index, "x": x, "y": y})

        # Create edges dataframe
        edges_data = []
        for edge in g.es:
            u, v = edge.source, edge.target
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Determine if edge is original (regular) or rewired
            ring_distance = min(abs(u - v), n - abs(u - v))
            edge_type = "Original" if ring_distance <= k // 2 else "Rewired"

            edges_data.append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "edge_type": edge_type}
            )

        return pd.DataFrame(nodes_data), pd.DataFrame(edges_data)


    # Create network at current p value
    current_network = create_network_at_p(current_p)
    nodes_df, edges_df = create_network_data(current_network, 300, 4)

    # Create network visualization
    if len(nodes_df) > 0:
        # Edges chart
        edges_chart = (
            alt.Chart(edges_df)
            .mark_rule(strokeWidth=1, opacity=0.6)
            .encode(
                x=alt.X("x1:Q", scale=alt.Scale(domain=[-1.2, 1.2]), axis=None),
                y=alt.Y("y1:Q", scale=alt.Scale(domain=[-1.2, 1.2]), axis=None),
                x2="x2:Q",
                y2="y2:Q",
                color=alt.Color(
                    "edge_type:N",
                    scale=alt.Scale(
                        domain=["Original", "Rewired"], range=["gray", "red"]
                    ),
                    legend=alt.Legend(title="Edge Type"),
                ),
            )
        )

        # Nodes chart
        nodes_chart = (
            alt.Chart(nodes_df)
            .mark_circle(size=30, stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-1.2, 1.2]), axis=None),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.2, 1.2]), axis=None),
                fill=alt.value("lightblue"),
            )
        )

        # Combine network
        network_chart = (
            (edges_chart + nodes_chart)
            .properties(
                width=350,
                height=350,
                title=f"Network Structure at p = {current_p:.3f}",
            )
            .resolve_scale(color="independent")
        )
    else:
        network_chart = alt.Chart().mark_text(
            text="Network visualization unavailable", fontSize=14
        )
    return (network_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create Clustering Coefficient Chart""")
    return


@app.cell(hide_code=True)
def _(alt, current_data, current_p, ws_data):
    # Create clustering coefficient chart
    if len(ws_data) > 0:
        clustering_chart = (
            alt.Chart(ws_data)
            .mark_line(strokeWidth=3, color="blue")
            .encode(
                x=alt.X(
                    "p:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Rewiring Probability (p)",
                ),
                y=alt.Y("clustering:Q", title="Clustering Coefficient"),
            )
        )

        # Current position marker
        if len(current_data) > 0:
            current_clustering_point = (
                alt.Chart(current_data)
                .mark_circle(size=200, color="blue", stroke="black", strokeWidth=2)
                .encode(
                    x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("clustering:Q"),
                )
            )
            clustering_chart = clustering_chart + current_clustering_point

        clustering_chart = clustering_chart.properties(
            width=400,
            height=200,
            title=f"Clustering Coefficient (Current C = {current_data.iloc[0]['clustering']:.3f} at p = {current_p:.3f})"
            if len(current_data) > 0
            else "Clustering Coefficient",
        )
    else:
        clustering_chart = alt.Chart().mark_text(
            text="Please implement the required functions to see the visualization",
            fontSize=16,
        )
    return (clustering_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create Average Path Length Chart""")
    return


@app.cell(hide_code=True)
def _(alt, current_data, current_p, ws_data):
    # Create average path length chart
    if len(ws_data) > 0:
        path_chart = (
            alt.Chart(ws_data)
            .mark_line(strokeWidth=3, color="red")
            .encode(
                x=alt.X(
                    "p:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Rewiring Probability (p)",
                ),
                y=alt.Y("path_length:Q", title="Average Path Length"),
            )
        )

        # Current position marker
        if len(current_data) > 0:
            current_path_point = (
                alt.Chart(current_data)
                .mark_circle(size=200, color="red", stroke="black", strokeWidth=2)
                .encode(
                    x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("path_length:Q"),
                )
            )
            path_chart = path_chart + current_path_point

        path_chart = path_chart.properties(
            width=400,
            height=200,
            title=f"Average Path Length (Current L = {current_data.iloc[0]['path_length']:.3f} at p = {current_p:.3f})"
            if len(current_data) > 0
            else "Average Path Length",
        )
    else:
        path_chart = alt.Chart().mark_text(
            text="Please implement the required functions to see the visualization",
            fontSize=16,
        )
    return (path_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create Small-World Coefficient Chart""")
    return


@app.cell(hide_code=True)
def _(alt, current_data, current_p, ws_data):
    # Create small-world coefficient chart
    if len(ws_data) > 0:
        sigma_chart = (
            alt.Chart(ws_data)
            .mark_line(strokeWidth=3, color="orange")
            .encode(
                x=alt.X(
                    "p:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Rewiring Probability (p)",
                ),
                y=alt.Y("small_world:Q", title="Small-World Coefficient σ"),
            )
        )

        if len(current_data) > 0 and "small_world" in current_data.columns:
            current_sigma_point = (
                alt.Chart(current_data)
                .mark_circle(
                    size=200, color="orange", stroke="black", strokeWidth=2
                )
                .encode(
                    x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("small_world:Q"),
                )
            )
            sigma_chart = sigma_chart + current_sigma_point

        sigma_chart = sigma_chart.properties(
            width=600,
            height=400,
            title=f"Small-World Coefficient (Current σ = {current_data.iloc[0]['small_world']:.2f} at p = {current_p:.3f})"
            if len(current_data) > 0
            else "Small-World Coefficient",
        )
    else:
        sigma_chart = alt.Chart().mark_text(
            text="Please implement the required functions to see the visualization",
            fontSize=16,
        )
    return (sigma_chart,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
