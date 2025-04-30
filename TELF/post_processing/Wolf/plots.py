import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import plotly.graph_objs as go
from .wolf import Wolf


def create_slice_graphs(X, node_ids=None, node_attributes=None, slice_ids=None, verbose=False):
    """
    Parses a tensor to create a graph for each slice. The results are stored as dictionary.
    If `index_map` is provided the keys of the dictionary use the map provided, otherwise the
    numerical indices are used.

    Parameters:
    -----------
    X: np.ndarray, sparse.COO
        An (m,m,n) tensor representing connections over n slices
    node_ids: dict
        A dictionary of m length where the keys are integers from 0-m and the values are 
        corresponding node ids. If None, node ids are just indices. Default is None.
    node_attributes: dict
        A dictionary of m length where keys are node ids and values are subdicts of attributes.
        If None, nodes do not have any attributes. Default is None.
    slice_ids: dict
        A dictionary of n length where the keys are integers from 0-n and values for the slicing
        dimension. If None, slices just use indices. Default is None.
    verbose: bool
        Show progress of computation

    Returns:
    --------
    dict: 
        A dictionary that stores graph slices formed from the tensor
    """
    # input check
    assert len(X.shape) == 3, "X is not a tensor!"
    assert X.shape[0] == X.shape[1], "Slices of X are not square!"
    assert X.shape[0] <= len(node_ids), f"Invalid node map for tensor of shape {X.shape}"
    if slice_ids is None:
        slice_ids = {}
    
    # setup Wolf
    wolf = Wolf(verbose=verbose)
    wolf.node_ids = node_ids
    wolf.attributes = node_attributes
    
    slices = {}
    for slice_index in tqdm(range(X.shape[-1]), disable = not verbose):
        S = X[:, :, slice_index]
        G = create_graph(S, wolf=wolf)
        slices[slice_ids.get(slice_index, slice_index)] = G
    return slices


def create_graph(X, wolf=None, node_ids=None, node_attributes=None, verbose=False):
    """
    Wrapper function to create a graph from a matrix.  
    
    A Wolf object can be passed to this function to be reused for creating multiple graphs.
    Otherwise a new Wolf object can be instantiated 

    Parameters:
    -----------
    X: np.ndarray, sparse.COO
        An (m,m,) matrix
    wolf: Wolf object, optional
        An instance of Wolf that can be reused multiple times to generate graphs using same node
        ids and attributes. Default is None.
    node_ids: dict
        A dictionary of m length where the keys are integers from 0-m and the values are 
        corresponding node ids. If None, node ids are just indices. Default is None.
    node_attributes: dict
        A dictionary of m length where keys are node ids and values are subdicts of attributes.
        If None, nodes do not have any attributes. Default is None.

    Returns:
    --------
    dict: 
        A dictionary that stores graph slices formed from the tensor
    """
    # input check
    assert len(X.shape) == 2, "X is not a matrix!"
    assert X.shape[0] == X.shape[1], "Slices of X are not square!"
    if wolf is None and node_ids is None:
        raise ValueError('Either existing Wolf object needs to be passed as `wolf` or `node_ids` need ' \
                         'to be passed to instantiate a new Wolf object')
    
    if wolf is not None and node_ids is not None:
        raise ValueError('An existing Wolf object was passed along with `node_ids`. Either one or the other ' \
                         'should be supplied to this function, but not both.')
                
    if wolf is None:
        wolf = Wolf(verbose=verbose)
        wolf.node_ids = node_ids
        wolf.attributes = node_attributes
    return wolf.create_graph(X, use_weighted_value=True).G


def find_connected_subgraph(G, target_node):
    """
    Return a set of all nodes in the same connected component as the 
    target node in an undirected graph.

    Parameters:
    -----------
    G: networkx.DiGraph)
        The input graph.
    target_node: 
        The node whose connected component is to be found.

    Returns:
    --------
    set: 
        A set of nodes in the same connected component as the target node.

    Raises:
    -------
    ValueError: 
        If the target_node is not in G.
    """
    if target_node not in G:
        raise ValueError(f"Target node {target_node} not found in the graph.")

    undirected_G = G.to_undirected()
    return nx.node_connected_component(undirected_G, target_node)


def compute_size_map(G, min_node_size, max_node_size, max_degree=None):
    """
    Computes a mapping of node sizes for a given graph based on their degrees.

    The function calculates node sizes for visualization by linearly interpolating 
    between `min_node_size` and `max_node_size` based on the degree of each node. Nodes 
    with the lowest degree get `min_node_size`, and those with the highest degree 
    (or `max_degree` if specified) get `max_node_size`. If the provided `max_degree` is 
    exceeded by a node in `G` then the user supplied `max_degree` is overruled. 

    Parameters
    ----------
    G: networkx.Graph
        The graph for which node sizes are to be computed. Each node's degree is used 
        in the computation.
    min_node_size: int
        The minimum size assigned to a node in the graph. Represents the size for the 
        node(s) with the lowest degree.
    max_node_size: int
        The maximum size assigned to a node in the graph. Represents the size for the 
        node(s) with the highest degree.
    max_degree: int, optional
        The maximum degree to consider for node size scaling. If not provided, it is 
        set to the highest degree in the graph. Default is None.

    Returns
    -------
    dict
        A dictionary mapping each node in the graph to its computed size. The sizes are 
        calculated based on the degree of the nodes, linearly scaled between min_node_size 
        and max_node_size.
    """
    degrees = dict(G.degree())
    min_degree = min(degrees.values())
    if not max_degree:
        max_degree = 0
    max_degree = max(max_degree, max(degrees.values()))
        
    if max_degree == min_degree:
        normalized_sizes = [max_node_size for _ in degrees]
    else:
        normalized_sizes = [
            min_node_size + (degrees[node] - min_degree) * (max_node_size - min_node_size) / (max_degree - min_degree)
            for node in G.nodes()
        ]
    return {node: size for node, size in zip(G.nodes(), normalized_sizes)}


def compose_graphs(slices, window, verbose=False):
    """
    Compose graphs from a dictionary based on a sliding window.

    Parameters:
    -----------
    slices: dict
        A dictionary where keys are slices and the values are NetworkX graph objects.
    window: int
        The size of the sliding window for composing graphs. If window == 1, no composition is done. 
          If window is greater than the length of the graph list or is 0, each graph is composed with 
          all preceding graphs.

    Returns:
    --------
    dict: 
        A dictionary with the same keys as slices, where each value is a composed graph according to the 
        window size.
    """
    assert isinstance(window, int) and window >= 0, "`window` must be an int greater than or equal to 0"
    if window == 0:
        window = len(slices) + 1

    # preserve order of keys as in original dict
    keys = list(slices.keys()) 

    composed_graphs = {}
    for i, slice_id in tqdm(enumerate(keys), total=len(keys), disable=not verbose):
        
        # determine range of indices for graphs to be composed
        if window == 1 or len(slices) == 1:
            composed_graph = slices[slice_id]
        elif window > len(slices):
            start_index = 0
        else:
            start_index = max(0, i - window + 1)

        # compose the graphs
        if window != 1 and len(slices) != 1:
            graphs_to_compose = [slices[keys[j]] for j in range(start_index, i + 1)]
            composed_graph = nx.compose_all(graphs_to_compose)

        composed_graphs[slice_id] = composed_graph

    return composed_graphs


def create_neighbor_dict(graph, target_nodes):
    """
    Create a dictionary where each key is a node from the node_list and
    the value is a set of nodes that the key node is connected to in the graph.

    Parameters:
    -----------
    graph: networkx.Graph
        The NetworkX graph object.
    target_nodes: list
        The list of nodes for which neighbors are to be found.

    Returns:
    --------
    dict:
        A dictionary with nodes as keys and sets of neighbors as values.
    """
    neighbor_dict = {}
    for node in target_nodes:
        if node in graph:
            neighbor_dict[node] = set(graph.neighbors(node)) | {node}
    return neighbor_dict


def update_positions(graph, positions, center=(0.5, 0.5), scale=0.5):
    """
    Update the positions of nodes in a graph based on the spring layout algorithm.

    Parameters:
    -----------
    graph: networkx.Graph
        The NetworkX graph for which the node positions need to be updated. 
        This graph should contain all the nodes referenced in 'positions'.
    positions: dict
        A dictionary with nodes as keys and positions as values. The positions are used as 
        initial positions for the spring layout algorithm. The format should be {node: (x, y), ...}, 
        where 'x' and 'y' are the coordinates of the node.
    center: tuple, optional
        A tuple representing the center of the layout (default is (0.5, 0.5)). This parameter
        scales the final positions around this center point.
    scale: float, optional
        A scaling factor for the layout (default is 0.5). This parameter scales the positions 
        of the nodes relative to the center. A larger scale will lead to a more spread out layout.

    Returns:
    --------
    new_positions : dict
        A dictionary containing the updated positions of the nodes after applying the spring layout algorithm. 
        The format is similar to the input 'positions' dictionary: {node: (x, y), ...}.
    """
    new_positions = nx.spring_layout(graph, pos=positions, center=center, scale=scale)
    return new_positions


def create_hover_attributes(G, nodes, hover_attributes, directed=False, show_key=False):
    """
    Generate hover text for nodes in a graph visualization.

    This function creates a list of strings containing HTML-formatted text. Each string corresponds to a node 
    in the graph and includes specified attributes to be displayed when hovering over the node in a visualization.

    Parameters:
    -----------
    G: networkx.Graph
        The NetworkX graph object from which node attributes are retrieved.
    nodes: iterable
        An iterable of nodes for which hover text is to be generated. Each node must exist in the graph G.
    hover_attributes: list
        A list of strings representing the names of the attributes to be included in the hover text. 
        If an attribute is not found for a node, it is replaced with 'Unknown'.
    directed: boolean
        Whether the graph is directed and edge weights should be recorded as attributes. Default is False.
    show_key: boolean
        If True, attribute name (the attribute key) is included in the hover text. Default is False. 

    Returns:
    --------
    list
        A list of HTML-formatted strings. Each string corresponds to the hover text for a node, containing 
        the specified attributes and the node identifier itself.

    Notes:
    ------
    - The first attribute in the hover_attributes list is bolded in the hover text.
    - Each attribute value is separated by a line break (<br>) in the hover text.
    - The node's identifier is always included as the last item in the hover text.
    """
    if directed:
        incoming_edge_weights = {node: 0 for node in G.nodes()}
        outgoing_edge_weights = {node: 0 for node in G.nodes()}
        for u, v, data in G.edges(data=True):
            outgoing_edge_weights[u] += data['weight']
            incoming_edge_weights[v] += data['weight']
    
    output = []
    for node in nodes:
        text_list = []
        for attribute in hover_attributes:
            if show_key:
                text = f"{attribute}: {G.nodes[node].get(attribute, 'Unknown')}"
            else:
                text = f"{G.nodes[node].get(attribute, 'Unknown')}"
            text_list.append(text)
        if directed:
            text_list.append(f"Outgoing Edge Sum: {outgoing_edge_weights[node]:.3f}")
            text_list.append(f"Incoming Edge Sum: {incoming_edge_weights[node]:.3f}")
        text_list.insert(0, str(node))
        text_list[0] = f"<b>{text_list[0]}</b>"
        output.append('<br>'.join(text_list))
    return output
                
                      
def create_frame(G, pos, target_nodes, highlight_nodes, size_map, hover_attributes, 
                 filter_isolated_nodes, show_key=True):
    """
    Create Plotly traces for a single frame (graph).
    
    Edge traces are created to represent graph connections using lines. Node traces include all 
    nodes in the graph with specified hover information and marker properties. Each target node 
    generates a separate trace to visualize its associated subgraph nodes. Highlighted nodes are 
    given a distinct styling for easy identification.

    Parameters:
    -----------
    G: networkx.Graph
        The NetworkX graph object to be visualized.
    pos: dict
        A dictionary with nodes as keys and positions as values. The positions are 
        used to place the nodes in the plot.
    target_nodes: dict
        A dictionary where each key is a node in the graph and the value is a set of 
        nodes that are related to the key node (e.g., neighbors).
    highlight_nodes: list
        A list of nodes to be highlighted in the visualization.
    size_map: dict
        A dictionary mapping each node to its size in the visualization. Used to 
        vary the size of nodes in the plot.
    hover_attributes: list
        A list of node attributes to be displayed when hovering over nodes in the plot.
    filter_isolated_nodes: bool
        If True, nodes with no edges are excluded from the plot unless they are in `target_nodes`.
    show_key: boolean
        If True, attribute name (the attribute key) is included in the hover text.

    Returns:
    --------
    list
        A list of Plotly Scatter objects (traces) representing the edges, nodes, and 
        target nodes in the graph. This includes:
        - A trace for all edges in the graph.
        - A trace for all nodes in the graph.
        - Individual traces for each group of target nodes.
    """
    
    # node filtering logic based on filter_isolated_nodes
    def should_include_node(node):
        return not filter_isolated_nodes or len(G.edges(node)) > 0 or node in target_nodes

    # first create the list of nodes to plot
    nodes_to_plot = {node for node in G.nodes if should_include_node(node)}

    # edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges:
        if edge[0] in nodes_to_plot and edge[1] in nodes_to_plot:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),  #edge color is a shade of gray
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )

    # node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in nodes_to_plot],
        y=[pos[node][1] for node in nodes_to_plot],
        mode='markers',
        hoverinfo='text',
        text=create_hover_attributes(G, nodes_to_plot, hover_attributes, show_key=show_key),
        marker=dict(
            size=[size_map[node] for node in nodes_to_plot],
            line_width=[3 if node in highlight_nodes else 1 for node in nodes_to_plot],
            line_color=['black' if node in highlight_nodes else 'DarkSlateGrey' for node in nodes_to_plot],
        ),
        showlegend=False
    )
    
    # next, create traces for target_nodes
    target_nodes_traces = []
    for key_node, subgraph_nodes in target_nodes.items():
        if not hover_attributes:
            trace_name = key_node
        else:
            trace_name = str(G.nodes[key_node].get(hover_attributes[0], key_node))
            
        subgraph_node_x = [pos[node][0] for node in subgraph_nodes]
        subgraph_node_y = [pos[node][1] for node in subgraph_nodes]
        subgraph_node_trace = go.Scatter(
            x=subgraph_node_x, y=subgraph_node_y,
            mode='markers',
            hoverinfo='text',
            text=create_hover_attributes(G, subgraph_nodes, hover_attributes, show_key=show_key),
            marker=dict(
                size=[size_map[node] for node in subgraph_nodes],
                line_width = [3 if node in highlight_nodes else 1 for node in G],
                line_color = ['black' if node in highlight_nodes else 'DarkSlateGrey' for node in G],
            ),
            name=trace_name,
            showlegend=True
        )
        target_nodes_traces.append(subgraph_node_trace)
    
    return [edge_trace, node_trace] + target_nodes_traces


def create_directed_frame(G, pos, size_map, hover_attributes, filter_isolated_nodes):
    """
    Create Plotly traces for a single frame (directed graph).
    
    These traces visualize a directed graph with nodes and edges displayed with specific styling 
    and color coding based on edge weights. The nodes are colored based on the proportion of their incoming 
    and outgoing edge weights, with a colorscale from blue (outgoing) to red (incoming). Edges are visually 
    split into two segments: blue for the outgoing part and red for the incoming part to indicate direction.

    Parameters:
    -----------
    G: networkx.DiGraph
        The directed graph to visualize.
    pos: dict
        A dictionary specifying the position of each node (e.g., output of networkx's layout functions).
    size_map: dict
        A dictionary mapping each node to its size in the visualization. Used to 
        vary the size of nodes in the plot.
    hover_attributes: list
        A list of node attributes to be displayed when hovering over nodes in the plot.
    filter_isolated_nodes: bool
        If True, nodes with no edges are excluded from the plot

    Returns:
    --------
    list
        list of three plotly.graph_objs.Scatter
          1. The first Scatter trace represents the first half of each edge (outgoing/blue).
          2. The second Scatter trace represents the second half of each edge (incoming/red).
          3. The node trace with colors reflecting the balance between incoming and outgoing weights and sizes 
             based on `size_map`.
    """
    # init dictionaries to store the sum of incoming and outgoing weights
    incoming_weights = {node: 0 for node in G.nodes()}
    outgoing_weights = {node: 0 for node in G.nodes()}

    # calculate the weights
    for u, v, data in G.edges(data=True):
        outgoing_weights[u] += data['weight']
        incoming_weights[v] += data['weight']

    # node filtering logic based on filter_isolated_nodes
    def should_include_node(node):
        return not filter_isolated_nodes or len(G.edges(node)) > 0
    nodes_to_plot = {node for node in G.nodes if should_include_node(node)}
        
    # prepare node colors based on the ratio of incoming to total weight
    node_color_values = []
    for node in nodes_to_plot:
        total_weight = incoming_weights[node] + outgoing_weights[node]
        if total_weight > 0:
            proportion_incoming = incoming_weights[node] / total_weight
        else:  # if no edges, set to neutral color (purple)
            proportion_incoming = 0.5

        # store the proportion value, which will be mapped to a colorscale
        node_color_values.append(proportion_incoming)

    # edge traces
    # edges go blue to red
    edge_x = []
    edge_y = []
    edge_x_half = []
    edge_y_half = []
    for edge in G.edges():
        if edge[0] in nodes_to_plot and edge[1] in nodes_to_plot:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            edge_x.extend([x0, xm, None])
            edge_y.extend([y0, ym, None])
            edge_x_half.extend([xm, x1, None])
            edge_y_half.extend([ym, y1, None])

    edge_trace_1 = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    edge_trace_2 = go.Scatter(
        x=edge_x_half,
        y=edge_y_half,
        line=dict(width=0.5, color='red'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # node trace
    # nodes are colored using map defined above
    node_trace = go.Scatter(
        x=[pos[node][0] for node in nodes_to_plot],
        y=[pos[node][1] for node in nodes_to_plot],
        text=create_hover_attributes(G, nodes_to_plot, hover_attributes, directed=True),
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=[size_map[node] for node in nodes_to_plot],
            color=node_color_values,  # Use the numeric scale for colors
            colorbar=dict(title='Node Color:<br>Proportion of Incoming/<br>Outgoing Edges'),
            colorscale='Bluered',  # Use a reversed Blue-Red color scale
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        showlegend=False
    )
    return [edge_trace_1, edge_trace_2, node_trace]



def plot_tensor_graph_slices(X, node_ids, slice_ids, *, node_attributes=None, trace_nodes=[], window=1, 
                             globalize_node_sizes=True, highlight_nodes=[], filter_isolated_nodes=False, 
                             title='', width=900, height=900, max_node_size=50, min_node_size=3, verbose=False):
    """
    Visualize slices of a tensor graph as an animated sequence using Plotly.
    
    This function flattens the tensor along the third dimension to produce cumulative graph information.
    Each slice of the tensor represents an adjaceny matrix of a graph, which can be visualized by this function.
    The third dimension of the tensor represents some connection between the graphs (i.e. time) so this method
    allows for a sliding window to be set that controls how many sequential graphs should be connected for each
    slice of the visualuzation. The outout of the function is an animated Plotly figure which includes a 
    slider for navigating through different slices.

    TODO: perform some quality checks on the tensor X + node_ids + slice_ids

    Parameters:
    -----------
    X: numpy.ndarray
        A 3D tensor representing a series of adjacency matrices for graph slices.
    node_ids: dict
        A dictionary mapping node identifiers to indices in the tensor.
    slice_ids: dict
        A dictionary mapping slice identifiers to indices in the tensor.
    node_attributes: dict, optional
        A dictionary of dictionaries containing attributes for each node.
    trace_nodes: list, optional
        A list of nodes to trace throughout the animation.
    window: int, optional
        The size of the sliding window for composing graph slices (Default is 1).
    globalize_node_sizes: bool, optional
        If True, node sizes are normalized globally across all slices (Default is True).
    highlight_nodes: list, optional
        A list of nodes to be highlighted in the visualization. This is different from trace_nodes. Nodes included
        in trace_nodes will be given a distinct color and label in the plotly figure. These are interactive and can
        be enabled/disabled. Nodes in higlight_nodes are simply given a thick black border in the plot.
    filter_isolated_nodes: bool, optional
        If True, nodes with no edges are excluded from the plot unless they are in `trace_nodes`.
        (Default is True).
    title: str, optional
        Title of the plot.
    width: int, optional
        Width of the plot. (Default is 900).
    height: int, optional
        Height of the plot (Default is 900).
    max_node_size: int, optional
        Maximum node size in the plot (Default is 50).
    min_node_size: int, optional
        Minimum node size in the plot (Default is 3).
    verbose: bool, optional
        If True, prints outs progress during execution (Default is False).

    Returns:
    --------
    fig: plotly.graph_objects.Figure
        The Plotly figure object that can be displayed or exported

    Notes:
    ------
    """
    # flatten tensor on the third dimension to produce cumulative graph information
    X_flat = X.sum(axis=2)
    
    # set a global node size value if requested
    max_degree = None
    if globalize_node_sizes:
        _, counts = np.unique(X_flat.coords[0], return_counts=True)
        max_degree = counts[np.argmax(counts)]
    
    # sort the trace_nodes in increasing size so that more prominent nodes are rendered on top 
    if len(trace_nodes) > 1:
        node_ids_r = {v:k for k,v in node_ids.items()}
        trace_nodes = sorted(trace_nodes, key = lambda x: np.count_nonzero(X_flat.coords[0] == node_ids_r[x]))
    
    # create graphs for each slice
    if verbose:
        print('Generating graphs for each slice. . .', file = sys.stderr)
    graphs = create_slice_graphs(X, node_ids, node_attributes, slice_ids, verbose)
    
    # compose the graphs based on the sliding window
    if verbose:
        print(f'Composing graphs with window size={window}. . .', file = sys.stderr)
    composed_graphs = compose_graphs(graphs, window, verbose)

    # get the list of attributes to display with node
    hover_attributes = []
    if node_attributes:
        hover_attributes = list(node_attributes[next(iter(node_attributes))].keys())
    
    # create frames for each year
    frames = []
    pos = None
    if verbose:
        print(f'Generating Plotly frames. . .', file = sys.stderr)
    for i, (slice_id, g) in tqdm(enumerate(composed_graphs.items()), total=len(composed_graphs), disable=not verbose):
        if pos is None:
            pos = nx.circular_layout(g)
        pos = update_positions(g, pos)
        
        trace_nodes_map = create_neighbor_dict(g, trace_nodes)
        size_map = compute_size_map(g, min_node_size, max_node_size, max_degree=max_degree)
        traces = create_frame(g, pos, trace_nodes_map, highlight_nodes, size_map, hover_attributes, filter_isolated_nodes)
        frames.append(go.Frame(data=traces, name=slice_id))

    if verbose:
        print(f'Preparing figure. . .', file = sys.stderr)
        
    # Define the slider
    steps = []
    for slice_value in slice_ids.values():
        step = dict(
            method='animate',
            args=[[str(int(float(slice_value)))], {'frame': {'duration': 2000, 'redraw': True}, 'mode': 'immediate'}],
            label=str(int(float(slice_value)))
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'visible': True, 'prefix': 'Slice: '},
        pad={'t': 50},
        steps=steps
    )]

    # define the figure
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=title,
            showlegend=True,
            sliders=sliders,
            width=width,
            height=height,
            plot_bgcolor='rgba(0,0,0,0)',  # transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # transparent border
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            #annotations=annotations,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, 
                                    {"frame": {"duration": 1500, "redraw": True},
                                     "transition": {"duration": 1000}}])])],
            legend=dict(
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                ),
                itemsizing='constant'  # Ensure consistent size in the legend
            )
        ),
        frames=frames,
    )
    return fig


def plot_matrix_graph(X, node_ids, *, node_attributes=None, trace_nodes=[], highlight_nodes=[], 
                      filter_isolated_nodes=False, title='', width=900, height=900, max_node_size=50, 
                      min_node_size=3, show_key=False, verbose=False):
    """
    Visualize adjacency matrix of an undirected as an interactive Plotly scatter plot.

    Parameters:
    -----------
    X: numpy.ndarray
        A 2D  matrix representing an adjacency matrix
    node_ids: dict
        A dictionary mapping node identifiers to indices in the tensor.
    node_attributes: dict, optional
        A dictionary of dictionaries containing attributes for each node.
    trace_nodes: list, optional
        A list of nodes to trace throughout the animation.
    highlight_nodes: list, optional
        A list of nodes to be highlighted in the visualization. This is different from trace_nodes. Nodes included
        in trace_nodes will be given a distinct color and label in the plotly figure. These are interactive and can
        be enabled/disabled. Nodes in higlight_nodes are simply given a thick black border in the plot.
    filter_isolated_nodes: bool, optional
        If True, nodes with no edges are excluded from the plot unless they are in `trace_nodes`.
        (Default is True).
    title: str, optional
        Title of the plot.
    width: int, optional
        Width of the plot. (Default is 900).
    height: int, optional
        Height of the plot (Default is 900).
    max_node_size: int, optional
        Maximum node size in the plot (Default is 50).
    min_node_size: int, optional
        Minimum node size in the plot (Default is 3).
    show_key: boolean
        If True, attribute name (the attribute key) is included in the hover text. Default is False. 
    verbose: bool, optional
        If True, prints outs progress during execution (Default is False).

    Returns:
    --------
    fig: plotly.graph_objects.Figure
        The Plotly figure object that can be displayed or exported

    Notes:
    ------
    """
    
    # sort the trace_nodes in increasing size so that more prominent nodes are rendered on top 
    if len(trace_nodes) > 1:
        node_ids_r = {v:k for k,v in node_ids.items()}
        trace_nodes = sorted(trace_nodes, key = lambda x: np.count_nonzero(X.coords[0] == node_ids_r[x]))
    
    # create graphs for each slice
    if verbose:
        print('Generating graph. . .', file = sys.stderr)
    graph = create_graph(X, wolf=None, node_ids=node_ids, node_attributes=node_attributes)

    # get the list of attributes to display with node
    hover_attributes = []
    if node_attributes:
        hover_attributes = list(node_attributes[next(iter(node_attributes))].keys())
    
    # create trace 
    if verbose:
        print(f'Generating Plotly frame. . .', file = sys.stderr)
    pos = nx.spring_layout(graph)
    trace_nodes_map = create_neighbor_dict(graph, trace_nodes)
    size_map = compute_size_map(graph, min_node_size, max_node_size)
    traces = create_frame(graph, pos, trace_nodes_map, highlight_nodes, size_map, 
                          hover_attributes, filter_isolated_nodes, show_key)
    
    # define the figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            width=width,
            height=height,
            plot_bgcolor='rgba(0,0,0,0)',  # transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # transparent border
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                ),
                itemsizing='constant'  # Ensure consistent size in the legend
            )
        )
    )
    return fig


def plot_matrix_directed_graph(X, node_ids, *, node_attributes=None, filter_isolated_nodes=False, title='', 
                               width=900, height=900, heatmap=None, max_node_size=50, min_node_size=3, verbose=False):
    """
    Visualize adjacency matrix of a directed graph as an interactive Plotly scatter plot.

    Parameters:
    -----------
    X: numpy.ndarray
        A 2D  matrix representing an adjacency matrix
    node_ids: dict
        A dictionary mapping node identifiers to indices in the tensor.
    node_attributes: dict, optional
        A dictionary of dictionaries containing attributes for each node.
    filter_isolated_nodes: bool, optional
        If True, nodes with no edges are excluded from the plot unless they are in `trace_nodes`.
        (Default is True).
    title: str, optional
        Title of the plot.
    width: int, optional
        Width of the plot. (Default is 900).
    height: int, optional
        Height of the plot (Default is 900).
    max_node_size: int, optional
        Maximum node size in the plot (Default is 50).
    min_node_size: int, optional
        Minimum node size in the plot (Default is 3).
    verbose: bool, optional
        If True, prints outs progress during execution (Default is False).

    Returns:
    --------
    fig: plotly.graph_objects.Figure
        The Plotly figure object that can be displayed or exported

    Notes:
    ------
    """
    # create graphs for each slice
    if verbose:
        print('Generating graph. . .', file = sys.stderr)
    graph = create_graph(X, wolf=None, node_ids=node_ids, node_attributes=node_attributes)

    # get the list of attributes to display with node
    hover_attributes = []
    if node_attributes:
        hover_attributes = list(node_attributes[next(iter(node_attributes))].keys())
    
    # create trace 
    if verbose:
        print(f'Generating Plotly frame. . .', file = sys.stderr)
    pos = nx.spring_layout(graph)
    size_map = compute_size_map(graph, min_node_size, max_node_size)
    traces = create_directed_frame(graph, pos, size_map, hover_attributes, filter_isolated_nodes)
    
    # define the figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            width=width,
            height=height,
            plot_bgcolor='rgba(0,0,0,0)',  # transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # transparent border
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                ),
                itemsizing='constant'  # Ensure consistent size in the legend
            )
        )
    )
    return fig