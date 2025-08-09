import marimo

__generated_with = "0.14.13"
app = marimo.App()


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
    mo.accordion({
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
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 1: Global Clustering Coefficient

    The **global clustering coefficient** measures the overall tendency of nodes to cluster together. It's defined as:

    **C_global = (3 × number of triangles) / (number of connected triples)**

    Where:
    - A **triangle** is a set of 3 fully connected nodes
    - A **connected triple** is a set of 3 nodes where at least 2 edges exist

    In igraph, you can use `g.transitivity_undirected()` for the global clustering coefficient.
    """)
    return


@app.cell
def _(igraph):
    #Task 1
    @app.function
    def compute_global_clustering(g):
        """
        Compute the global clustering coefficient of a graph.
        
        Args:
            g (igraph.Graph): Input graph
            
        Returns:
            float: Global clustering coefficient (0.0 to 1.0)
        """
        # TODO: Implement the global clustering coefficient calculation
        # Hint: Use igraph's transitivity_undirected() method
        pass
    return (compute_global_clustering,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2: Average Path Length

    The **average path length** (also called **characteristic path length**) is the average number of edges in the shortest paths between all pairs of nodes.

    For a connected graph with N nodes:
    **L = (1 / (N(N-1))) × Σ d(i,j)**

    Where d(i,j) is the shortest path distance between nodes i and j.

    In igraph, you can use `g.distances()` to get all pairwise distances.
    """)
    return


@app.cell
def _(igraph, np):
    #Task 2
    @app.function
    def compute_average_path_length(g):
        """
        Compute the average shortest path length of a graph.
        
        Args:
            g (igraph.Graph): Input graph (should be connected)
            
        Returns:
            float: Average path length
        """
        # TODO: Implement the average path length calculation
        # Hint: Use g.distances() to get distance matrix, then compute mean
        # Be careful to exclude diagonal elements (distance from node to itself = 0)
        pass
    return (compute_average_path_length,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 3: Small-World Coefficient

    The **small-world coefficient** (σ) quantifies how "small-world" a network is by comparing it to equivalent random and regular networks:

    **σ = (C/C_random) / (L/L_random)**

    Where:
    - C = clustering coefficient of the network
    - C_random = clustering coefficient of equivalent random network  
    - L = average path length of the network
    - L_random = average path length of equivalent random network

    **σ >> 1** indicates small-world properties (high clustering but short paths like random graphs).

    For the random reference, use an Erdős–Rényi random graph with same number of nodes and edges.
    """)
    return


@app.cell
def _(igraph):
    #Task 3
    @app.function
    def compute_small_world_coefficient(g):
        """
        Compute the small-world coefficient using random graph as reference.
        
        Args:
            g (igraph.Graph): Input graph
            
        Returns:
            float: Small-world coefficient (σ)
        """
        # TODO: Implement the small-world coefficient calculation
        # 1. Compute C and L for the input graph
        # 2. Generate equivalent Erdős–Rényi random graph with same n and m
        # 3. Compute C_random and L_random for the random graph
        # 4. Return σ = (C/C_random) / (L/L_random)
        # 
        # Hint: Use igraph.Graph.Erdos_Renyi(n=n_nodes, m=n_edges) for random graph
        pass
    return (compute_small_world_coefficient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Interactive Visualization: Watts-Strogatz Model

    Now let's explore how these metrics change as we vary the rewiring probability in the Watts-Strogatz model!
    
    The **Watts-Strogatz model** starts with a ring lattice and randomly rewires edges with probability `p`:
    - `p = 0`: Regular ring lattice (high clustering, long paths)
    - `p = 1`: Random graph (low clustering, short paths)  
    - `0 < p < 1`: Small-world region (high clustering AND short paths)

    Use the slider below to explore different rewiring probabilities:
    """)
    return


@app.cell
def _(mo):
    # Rewiring probability slider
    p_slider = mo.ui.slider(
        start=0.0, 
        stop=1.0, 
        step=0.01, 
        value=0.1, 
        label="Rewiring Probability (p)"
    )
    p_slider
    return (p_slider,)


@app.cell
def _(alt, compute_average_path_length, compute_global_clustering, compute_small_world_coefficient, igraph, np, p_slider, pd):
    # Generate Watts-Strogatz networks and compute metrics
    def generate_ws_metrics(p_values, n=100, k=4):
        """Generate metrics for Watts-Strogatz networks across p values"""
        results = []
        
        for p in p_values:
            try:
                # Generate Watts-Strogatz graph
                g = igraph.Graph.Watts_Strogatz(dim=1, size=n, nei=k//2, p=p)
                
                # Ensure the graph is connected
                if not g.is_connected():
                    continue
                
                # Compute metrics using the student's functions
                C = compute_global_clustering(g)
                L = compute_average_path_length(g)
                sigma = compute_small_world_coefficient(g)
                
                results.append({
                    'p': p,
                    'clustering': C,
                    'path_length': L,
                    'small_world': sigma
                })
            except:
                # Skip problematic parameter values
                continue
                
        return pd.DataFrame(results)

    # Generate data for visualization
    p_values = np.logspace(-4, 0, 50)  # From 0.0001 to 1.0
    _ws_data = generate_ws_metrics(p_values)

    # Normalize metrics for comparison (relative to p=0 values)
    if len(_ws_data) > 0:
        _c0 = _ws_data.iloc[0]['clustering']  # clustering at p≈0
        _l0 = _ws_data.iloc[0]['path_length']  # path length at p≈0
        
        _ws_data['clustering_normalized'] = _ws_data['clustering'] / _c0
        _ws_data['path_length_normalized'] = _ws_data['path_length'] / _l0

    # Create current point data
    _current_p = p_slider.value
    try:
        _current_g = igraph.Graph.Watts_Strogatz(dim=1, size=100, nei=2, p=_current_p)
        if _current_g.is_connected():
            _current_c = compute_global_clustering(_current_g)
            _current_l = compute_average_path_length(_current_g)
            _current_sigma = compute_small_world_coefficient(_current_g)
            
            _current_data = pd.DataFrame([{
                'p': _current_p,
                'clustering_normalized': _current_c / _c0 if len(_ws_data) > 0 else _current_c,
                'path_length_normalized': _current_l / _l0 if len(_ws_data) > 0 else _current_l,
                'small_world': _current_sigma
            }])
        else:
            _current_data = pd.DataFrame()
    except:
        _current_data = pd.DataFrame()

    # Create interactive visualization
    if len(_ws_data) > 0:
        # Base chart for the curves
        _base = alt.Chart(_ws_data).add_selection(
            alt.selection_interval()
        ).transform_fold(
            ['clustering_normalized', 'path_length_normalized'],
            as_=['metric', 'value']
        )

        # Line chart
        _lines = _base.mark_line(strokeWidth=3).encode(
            x=alt.X('p:Q', scale=alt.Scale(type='log'), title='Rewiring Probability (p)'),
            y=alt.Y('value:Q', title='Normalized Value'),
            color=alt.Color('metric:N', 
                          scale=alt.Scale(domain=['clustering_normalized', 'path_length_normalized'],
                                        range=['blue', 'red']),
                          legend=alt.Legend(title="Metric",
                                          labelExpr="datum.value == 'clustering_normalized' ? 'Clustering C(p)/C(0)' : 'Path Length L(p)/L(0)'"))
        )

        # Current position marker
        if len(_current_data) > 0:
            _current_points = alt.Chart(_current_data).transform_fold(
                ['clustering_normalized', 'path_length_normalized'],
                as_=['metric', 'value']
            ).mark_circle(size=200, stroke='black', strokeWidth=2).encode(
                x=alt.X('p:Q', scale=alt.Scale(type='log')),
                y=alt.Y('value:Q'),
                color=alt.Color('metric:N', 
                              scale=alt.Scale(domain=['clustering_normalized', 'path_length_normalized'],
                                            range=['blue', 'red']),
                              legend=None)
            )
            
            _combined_chart = (_lines + _current_points).resolve_scale(
                color='independent'
            ).properties(
                width=600,
                height=400,
                title=f"Small-World Transition (Current p = {_current_p:.3f})"
            )
        else:
            _combined_chart = _lines.properties(
                width=600,
                height=400,
                title="Small-World Transition"
            )

        # Small-world coefficient chart
        _sigma_chart = alt.Chart(_ws_data).mark_line(strokeWidth=3, color='green').encode(
            x=alt.X('p:Q', scale=alt.Scale(type='log'), title='Rewiring Probability (p)'),
            y=alt.Y('small_world:Q', title='Small-World Coefficient σ')
        )

        if len(_current_data) > 0 and 'small_world' in _current_data.columns:
            _current_sigma_point = alt.Chart(_current_data).mark_circle(
                size=200, color='green', stroke='black', strokeWidth=2
            ).encode(
                x=alt.X('p:Q', scale=alt.Scale(type='log')),
                y=alt.Y('small_world:Q')
            )
            _sigma_chart = _sigma_chart + _current_sigma_point

        _sigma_chart = _sigma_chart.properties(
            width=600,
            height=300,
            title=f"Small-World Coefficient (Current σ = {_current_data.iloc[0]['small_world']:.2f} at p = {_current_p:.3f})" if len(_current_data) > 0 else "Small-World Coefficient"
        )

        # Combine charts vertically
        _final_chart = alt.vconcat(_combined_chart, _sigma_chart).resolve_scale(
            x='shared'
        )
        
        _final_chart
    else:
        alt.Chart().mark_text(text="Please implement the required functions to see the visualization", fontSize=16)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Visualization

    The interactive plot above shows three key insights about small-world networks:

    1. **Top Panel**: Shows how clustering coefficient C(p) and path length L(p) change with rewiring probability
       - **Blue line**: Clustering coefficient (normalized by C(0))
       - **Red line**: Average path length (normalized by L(0))

    2. **Bottom Panel**: Shows the small-world coefficient σ = (C/C_random)/(L/L_random)
       - **σ >> 1**: Strong small-world properties
       - **Peak around p ≈ 0.01-0.1**: Optimal small-world region

    3. **Interactive Elements**: 
       - Move the slider to see how your current choice of p affects all metrics
       - The black circles show your current position on the curves

    ### Key Observations:
    - At **p = 0**: High clustering, long paths (regular lattice)
    - At **p = 1**: Low clustering, short paths (random graph)  
    - At **intermediate p**: High clustering AND short paths (small-world!)

    The small-world regime occurs when clustering decreases slowly but path length drops rapidly as p increases.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()