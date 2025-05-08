import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib
from tqdm import tqdm
from wordcloud import WordCloud
import random
import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from scipy.spatial import ConvexHull
import logging
log = logging.getLogger(__name__)
from .file_system import check_path
from .data_structures import sum_dicts
from .maps import get_id_to_name
from .graphs import create_authors_graph


def plot_authors_graph(df, id_col='s2_author_ids', name_col='s2_authors', title='Co-Authors Graph',
                       width=900, height=900, max_node_size=50, min_node_size=3):
    G = create_authors_graph(df, id_col)
    pos = nx.spring_layout(G)  # position nodes using networkx's spring layout
    name_map = get_id_to_name(df, name_col, id_col)

    # global normalization for node sizes
    degrees = dict(G.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    if max_degree == min_degree:
        normalized_sizes = [max_node_size for _ in degrees]
    else:
        normalized_sizes = [
            min_node_size + (degrees[node] - min_degree) * (max_node_size - min_node_size) / (max_degree - min_degree)
            for node in G.nodes()
        ]
    size_map = {node: size for node, size in zip(G.nodes(), normalized_sizes)}

    # find connected components, sorted by size in descending order
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    # create traces for each component
    traces = []
    for i, component in enumerate(components):
        component_label = f"Component {i}"

        # edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges(component):
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
            legendgroup=component_label
        )
        
        node_x = [pos[node][0] for node in component]
        node_y = [pos[node][1] for node in component]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=[f"<b>{name_map.get(id_, 'Unknown')}</b><br>{id_}" for id_ in component],
            hoverinfo='text',
            marker=dict(
                size=[size_map[node] for node in component],
                line_width=2
            ),
            name=component_label,
            legendgroup=component_label,
            showlegend=True
        )

        traces.extend([edge_trace, node_trace])

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

def create_wordcloud_from_df(df, col, n=30, save_path=None, figsize=(600,600)):
    """
    Generate and display a word cloud from a specified DataFrame column.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the text data.
    col: str
        The column name in the DataFrame with the text data.
    n: int, optional
        The number of top words to consider for the word cloud. Default is 30.
    save_path: str, optional
        Path to save the word cloud image. If not provided, the image is displayed but not saved.
    figsize: (int, int), optional
        A tuple specifying the size of the displayed image in inches. Default is (600,600).
    
    Returns:
    --------
    None: 
        Displays the word cloud using matplotlib. If save_path is provided, the word cloud is saved to the 
        specified path and not displayed.
    """
    # calculate the frequencies
    tokens    = [x.split() for x in df[col].to_list() if not pd.isna(x)]
    top_words = [dict(Counter(x)) for x in tokens]
    top_words = sum_dicts(top_words, n=n)

    # generate the word cloud image
    figsize_x, figsize_y = figsize
    wc = WordCloud(background_color="white", width=figsize_x*100, height=figsize_y*100)
    wc.generate_from_frequencies(top_words)
    
    # display the word cloud using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def _cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}

    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    def reduced_cmap(step): return np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, %d%%, 0%%)" % random.randint(1, 100)


def _make_wordcloud(
    B1_k,
    top_n,
    words_map,
    max_words=None,
    mask=None,
    background_color="black",
    min_font_size=10,
    max_font_size=None,
    prefer_horizontal=0.95,
    relative_scaling=0.1,
    colormap=None,
    contour_width=0.0,      
):
    """
    Internal helper that returns a WordCloud for the top_n words in B1_k.

    Parameters
    ----------
    B1_k : np.ndarray
        Array (usually a row or column from your matrix).
    top_n : int
        Number of top words to plot based on magnitude in B1_k.
    words_map : dict
        Maps index -> actual word string.
    max_words : int, optional
        Maximum number of words in the word cloud.
    mask : np.ndarray, optional
        A shape mask array for the cloud (e.g. np.zeros((800, 800)) ).
    background_color : str, optional
        Cloud background color, default='black'.
    min_font_size : int, optional
        Minimum font size for the smallest word.
    max_font_size : int, optional
        Maximum font size for the largest word.
    prefer_horizontal : float, optional
        Ratio (0 to 1) indicating how often to layout text horizontally.
    relative_scaling : float, optional
        How much to shrink more frequent words relative to others.
    colormap : matplotlib.colors.Colormap, optional
        A custom colormap. If None, we'll use a default.

    Returns
    -------
    WordCloud
        A generated word cloud for the top_n words in B1_k.
    """

    if colormap is None:
        # Example: a custom tinted colormap
        # or just colormap = plt.get_cmap("RdBu")
        # Here we apply user-defined _cmap_map if desired
        base_cmap = plt.get_cmap("RdBu")
        colormap = _cmap_map(lambda x: x * 0.6, base_cmap)

    # pick top_n largest values in B1_k
    index = np.argpartition(B1_k, -top_n)[-top_n:]
    # Ensure words_map is aligned to vocab and W
    if len(words_map) != len(B1_k):
        words_map = {i: word for i, word in enumerate(words_map[:len(B1_k)])}

    # Pick top_n largest values in B1_k
    index = np.argpartition(B1_k, -top_n)[-top_n:]

    # Build word frequency dictionary
    w_data = {}
    for i in index:
        if i in words_map:  # Avoid KeyError
            w_data[words_map[i]] = float(B1_k[i])


    wordcloud = WordCloud(
        width=1600,
        height=1600,
        background_color=background_color,
        relative_scaling=relative_scaling,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        prefer_horizontal=prefer_horizontal,
        max_words=max_words,
        mask=mask,
        colormap=colormap,
        contour_width=contour_width  
    ).generate_from_frequencies(w_data)

    return wordcloud


def create_wordcloud(
    W,
    vocab,
    top_n=10,
    path="",
    verbose=False,
    max_words=None,
    mask=None,
    background_color="black",
    min_font_size=10,
    max_font_size=None,
    prefer_horizontal=0.95,
    relative_scaling=0.1,
    colormap=None,
    contour_width=0.0,
    filename_base=None,
    grid_dimension=None
):
    """
    Creates and saves a word cloud PDF for each column k in W, focusing on top_n words.

    Parameters
    ----------
    W : np.ndarray
        A matrix of shape (num_terms, num_topics).
    vocab : list or array
        The vocabulary list, vocab[i] is the term for row i.
    top_n : int
        Number of top words to show for each topic.
    path : str
        Output directory to save PDFs (e.g. 'results/').
    verbose : bool
        Whether to show a tqdm progress bar.
    max_words : int, optional
        Maximum words to display in the cloud. Default None (no explicit limit).
    mask : np.ndarray, optional
        Shape mask for the cloud.
    background_color : str, optional
        The color behind all words. Default='black'.
    min_font_size : int, optional
        Minimum font size for the smallest word. Default=10.
    max_font_size : int, optional
        Maximum font size for the largest word. Default=None.
    prefer_horizontal : float, optional
        Ratio of how often text is laid horizontally. Default=0.95.
    relative_scaling : float, optional
        Amount to shrink frequent words relative to others. Default=0.1.
    colormap : colormap, optional
        A matplotlib colormap object or name.

    Returns
    -------
    None
    """

    vocab = np.array(vocab)
    if W.shape[0] > len(vocab):
        print(f"[Warning] Truncating W to match vocab length: {len(vocab)}")
        W = W[:len(vocab), :]

    words_map = {ii: str(word) for ii, word in enumerate(vocab[:W.shape[0]])}
    K = W.shape[1]  # number of topics

    os.makedirs(path, exist_ok=True)

    if grid_dimension is None:
        # 🔵 Old behavior: save individual PDFs
        for k in tqdm(range(K), disable=not verbose):
            B1_k = W[:, k]
            fig = plt.figure(figsize=(15, 15), dpi=300, constrained_layout=False)

            wordcloud = _make_wordcloud(
                B1_k=B1_k,
                top_n=top_n,
                words_map=words_map,
                max_words=max_words,
                mask=mask,
                background_color=background_color,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                prefer_horizontal=prefer_horizontal,
                relative_scaling=relative_scaling,
                colormap=colormap,
                contour_width=contour_width
            )

            plt.xticks([])
            plt.yticks([])
            plt.imshow(wordcloud, interpolation="nearest", aspect="equal")
            plt.tight_layout()

            if filename_base is not None:
                pdf_path = os.path.join(path, f"{filename_base}.pdf")
            else:
                pdf_path = os.path.join(path, f"{k}.pdf")

            plt.savefig(pdf_path)
            plt.close()

    else:
        #  grid plot
        cols = grid_dimension
        rows = math.ceil(K / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), dpi=300)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        for idx, k in enumerate(range(K)):
            B1_k = W[:, k]

            wordcloud = _make_wordcloud(
                B1_k=B1_k,
                top_n=top_n,
                words_map=words_map,
                max_words=max_words,
                mask=mask,
                background_color=background_color,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                prefer_horizontal=prefer_horizontal,
                relative_scaling=relative_scaling,
                colormap=colormap,
                contour_width=contour_width
            )

            ax = axes[idx]
            ax.imshow(wordcloud, interpolation="nearest", aspect="equal")
            ax.axis('off')
            ax.set_title(f"Topic {k}", fontsize=10)

        # Hide any unused subplots
        for idx in range(K, rows * cols):
            axes[idx].axis('off')

        plt.tight_layout()
        pdf_path = os.path.join(path, "wordcloud_grid.pdf")
        plt.savefig(pdf_path)
        plt.close()

def word_cloud(words, probabilities, path, max_words=30, background_color='white', format='png', **kwargs):
    """
    Generates and saves word cloud images based on words and their corresponding probabilities.

    Parameters:
    -----------
    words: list of list of str
        A list of lists, each containing words to be included in the word clouds.
    probabilities: list of list of float
        A list of lists, each containing probabilities corresponding to the words.
    path: str
        The directory path where the word cloud images will be saved.
    max_words: int, optional
        The maximum number of words to include in the word cloud (default is 30).
    background_color : str, optional
        Background color for the word cloud image (default is 'white').
    format: str, optional
        The format of the output image file (e.g., 'png', 'jpeg') (default is 'png').
    **kwargs:
        Additional keyword arguments to pass to the WordCloud constructor.

    Returns:
    --------
	None
    """
    if not all(len(words_row) == len(probs_row) for words_row, probs_row in zip(words, probabilities)):
        raise ValueError("Length of words and probabilities does not match in all rows")
        
    top_words = [{word: prob for word, prob in zip(words_row, probs_row)} for words_row, probs_row in zip(words.T, probabilities.T)]
    
    for k, word_dict in enumerate(top_words):
        wordcloud = WordCloud(max_words=max_words, background_color=background_color, **kwargs).fit_words(word_dict)
        save_dir = check_path(os.path.join(path, f'{k}'))
        wordcloud.to_file(os.path.join(save_dir, f'wordcloud_{k}.{format}'))

def plot_H_clustering(H, name="filename"):
    """
    Plots the centroids of the H-clusters

    Parameters
    ----------
    H: np.ndarray or scipy.sparse.csr_matrix
        H matrix from NMF
    name: File name to save

    Returns
    -------
    Matplotlib plots
    """
    labels = np.argmax(H, axis=0)
    uniques = np.unique(labels)
    for i, l in enumerate(uniques):
        fig = plt.figure(figsize=(6, 3*uniques.shape[0]))
        cluster = H[:, labels == l]
        cluster_means = np.mean(cluster, axis=1)
        cluster_stds = np.std(cluster, axis=1)
        ax = plt.subplot(uniques.shape[0], 1, i+1)
        plt.bar(np.arange(H.shape[0]), cluster_means, color=(0.2, 0.4, 0.6, 0.6))
        plt.title(f'{name} cluster {i}')
        plt.tight_layout()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{name}_{i}.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
        plt.close()

    return fig

def plot_umap(
    coords: np.ndarray,
    labels: list,
    output_path: Path,
    label_column: str,
    model_name: str,
    accepted_mask: Optional[np.ndarray] = None
) -> None:
    """
    Save a UMAP scatterplot with optional accepted-hull overlay.

    Parameters
    ----------
    coords : np.ndarray
        2D UMAP coordinates (n_samples, 2).
    labels : list
        Original labels corresponding to each coordinate.
    output_path : Path
        Filepath to save the resulting plot.
    label_column : str
        Name of the label column for legend entries.
    model_name : str
        Embedding model identifier for plot title.
    accepted_mask : Optional[np.ndarray]
        Boolean mask for accepted points; if provided, draws convex hull.
    """
    uniq = sorted(set(labels))
    color_map = {v: to_hex(get_cmap("tab20")(i % 20)) for i, v in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=[color_map[v] for v in labels],
               s=25, alpha=0.85)

    if accepted_mask is not None:
        accepted = coords[accepted_mask]
        if accepted.shape[0] >= 3:
            hull = ConvexHull(accepted)
            verts = accepted[hull.vertices]
            verts = np.vstack([verts, verts[0]])
            ax.fill(verts[:, 0], verts[:, 1],
                    facecolor="none", edgecolor="green", lw=2, alpha=0.8,
                    label="accepted hull")

    handles = [plt.Line2D([], [], marker="o", ls="", color=color_map[v]) for v in uniq]
    labels_legend = [f"{label_column}={v}" for v in uniq]
    if accepted_mask is not None:
        handles.append(plt.Line2D([], [], color="green", lw=2))
        labels_legend.append("accepted hull")

    ax.legend(handles, labels_legend, fontsize=8, loc="upper right")
    ax.set(xticks=[], yticks=[], title=f"UMAP – {model_name} embeddings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    log.info("Saved UMAP plot to %s", output_path)
