from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

def run_query():
    """
    Runs a Neo4j query that finds the 'estoppel' BOW_Vocab node,
    its connected nodes, any linked Keywords, and their Topics.
    """
    query = """
    MATCH (vocab:BOW_Vocab {id: 'estoppel'})--(connectedNode)
    OPTIONAL MATCH (connectedNode)--(kw:Keyword)--(topic:Topic_ID)
    WHERE kw.id CONTAINS 'estoppel'
    RETURN vocab, connectedNode, kw, topic
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(query)
        records = []
        for record in result:
            records.append({
                'vocab': record['vocab'],
                'connectedNode': record['connectedNode'],
                'kw': record.get('kw'),
                'topic': record.get('topic')
            })
    driver.close()
    return records


def build_full_graph(data):
    """
    Builds a full NetworkX graph from the Neo4j data.
    Each node’s ID is prefixed by its type so that 
    'BOW_estoppel' and 'KW_estoppel' remain distinct.
    """
    G = nx.Graph()
    for row in data:
        # BOW_Vocab node
        vocab_raw = row['vocab']
        bow_node_id = f"BOW_{vocab_raw['id']}"
        G.add_node(bow_node_id, label="BOW_estoppel")

        # Connected node
        if row['connectedNode']:
            c_raw = row['connectedNode']
            c_label = list(c_raw.labels)[0] if c_raw.labels else 'Unknown'
            c_id = f"{c_label}_{c_raw['id']}"
            G.add_node(c_id, label=c_label)
            G.add_edge(bow_node_id, c_id)

            # Keyword node
            if row['kw']:
                kw_raw = row['kw']
                kw_node_id = f"KW_{kw_raw['id']}"
                G.add_node(kw_node_id, label="KW_estoppel")
                G.add_edge(c_id, kw_node_id)

                # Topic node
                if row['topic']:
                    t_raw = row['topic']
                    t_id = f"Topic_{t_raw['id']}"
                    t_label = t_raw.get('label', t_raw['id'])
                    G.add_node(t_id, label="Topic_ID", text=t_label)
                    G.add_edge(kw_node_id, t_id)
    return G


def build_custom_aggregated_graph(G, center_BOW="BOW_estoppel", center_KW="KW_estoppel"):
    """
    For each center (BOW and Keyword), group immediate neighbors by label.
    Creates aggregator nodes like 'BOW_Appeals_Case' and 'KW_Appeals_Case'.
    The edge from center→aggregator has weight = count of neighbors.
    Overlap between BOW and KW aggregator nodes is an edge with weight = shared count.
    Additionally, an extra edge is added between the two centers with the total shared value,
    and a "total_difference" is computed as the absolute difference between the total BOW and KW aggregated counts.
    """
    H = nx.Graph()
    if center_BOW in G:
        H.add_node(center_BOW, type="center")
    if center_KW in G:
        H.add_node(center_KW, type="center")
        
    agg_BOW = defaultdict(set)
    agg_KW  = defaultdict(set)
    
    if center_BOW in G:
        for nbr in G.neighbors(center_BOW):
            if nbr in [center_BOW, center_KW]:
                continue
            lbl = G.nodes[nbr].get("label", "Unknown")
            agg_BOW[lbl].add(nbr)
    
    if center_KW in G:
        for nbr in G.neighbors(center_KW):
            if nbr in [center_BOW, center_KW]:
                continue
            lbl = G.nodes[nbr].get("label", "Unknown")
            agg_KW[lbl].add(nbr)
    
    total_shared = 0
    total_BOW_count = 0
    total_KW_count = 0
    
    all_labels = set(agg_BOW.keys()) | set(agg_KW.keys())
    for lbl in all_labels:
        count_bow = len(agg_BOW[lbl])
        count_kw  = len(agg_KW[lbl])
        total_BOW_count += count_bow
        total_KW_count += count_kw
        
        node_bow = None
        if count_bow > 0:
            node_bow = f"BOW_{lbl}"
            H.add_node(node_bow, type="agg", side="BOW", category=lbl)
            H.add_edge(center_BOW, node_bow, weight=count_bow)
        
        node_kw = None
        if count_kw > 0:
            node_kw = f"KW_{lbl}"
            H.add_node(node_kw, type="agg", side="KW", category=lbl)
            H.add_edge(center_KW, node_kw, weight=count_kw)
        
        if node_bow and node_kw:
            overlap = len(agg_BOW[lbl] & agg_KW[lbl])
            total_shared += overlap
            H.add_edge(node_bow, node_kw, weight=overlap, shared=True)
    
    total_difference = abs(total_BOW_count - total_KW_count)
    if center_BOW in H and center_KW in H:
        H.add_edge(center_BOW, center_KW, weight=total_shared, total_shared=True, total_difference=total_difference)
    
    return H


def add_invisible_node(H):
    """
    Adds an invisible node (spacer) at y = 0.27 that is not drawn and not in the legend.
    """
    H.add_node("invisible_spacer", type="invisible")
    manual_positions["invisible_spacer"] = (0, 0.27)


manual_positions = {
    "BOW_estoppel": (-5, 0),
    "KW_estoppel": (5, 0),
    "BOW_Statute": (-1.5, 0.05),
    "KW_Statute": (1.5, 0.05),
    "BOW_Supreme_Case": (-3, 0.1),
    "KW_Supreme_Case": (1.5, 0.1),
    "BOW_Appeals_Case": (-4.5, 0.15),
    "KW_Appeals_Case": (1.5, 0.15),
    "KW_Topic_ID": (5, 0.15),
}


def manual_layout(H, scale_y=0.5):
    pos = {}
    for node in H.nodes:
        if node in manual_positions:
            x, y = manual_positions[node]
            pos[node] = (x, y * scale_y)
    
    unplaced = [n for n in H.nodes if n not in pos]
    if unplaced:
        subgraph = H.subgraph(unplaced).copy()
        fallback_pos = nx.spring_layout(subgraph, k=0.5, iterations=30)
        for n in unplaced:
            x, y = fallback_pos[n]
            pos[n] = (x, y * scale_y)
    
    return pos


def get_color_for_category(cat):
    color_map = {
        "Supreme_Case": "#FF914D",
        "Appeals_Case": "#FFD45D",
        "Statute": "#74CA4D",
        "Topic_ID": "#36C9C6",
        "Unknown": "#808080"
    }
    return color_map.get(cat, "#808080")


def visualize_custom_aggregated_graph(H, figsize=(12,9)):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14

    pos = manual_layout(H, scale_y=0.5)
    plt.figure(figsize=figsize)
    
    # Filter out invisible nodes and edges
    visible_nodes = [n for n in H.nodes if H.nodes[n].get("type") != "invisible"]
    visible_edges = [(u, v) for u, v in H.edges if H.nodes[u].get("type") != "invisible" and H.nodes[v].get("type") != "invisible"]
    
    nx.draw_networkx_edges(H, pos, edgelist=visible_edges, edge_color='gray', width=1.5)
    
    node_colors = {}
    node_sizes = {}
    labels = {}
    
    for node in visible_nodes:
        data = H.nodes[node]
        if data.get("type") == "center":
            if node == "BOW_estoppel":
                node_colors[node] = "#5A2BB2"
            elif node == "KW_estoppel":
                node_colors[node] = "red"
            else:
                node_colors[node] = "#808080"
            node_sizes[node] = 8000
            labels[node] = node.replace("_", "\n").replace("ID", "(HNMFk)")
        else:
            cat = data.get("category", "Unknown")
            node_colors[node] = get_color_for_category(cat)
            node_sizes[node] = 8000
            labels[node] = cat.replace("_", "\n").replace("ID", "(HNMFk)")
    
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=visible_nodes,
        node_color=[node_colors[n] for n in visible_nodes],
        node_size=[node_sizes[n] for n in visible_nodes]
    )
    
    nx.draw_networkx_labels(H, pos, labels, font_size=16, font_weight="bold")
    
    shared_edges = {}
    other_edges = {}
    center_shared_edge = {}
    
    for u, v in visible_edges:
        w = H[u][v].get("weight", 0)
        if H[u][v].get("shared"):
            shared_edges[(u, v)] = f"shared: {w}"
        elif H[u][v].get("total_shared"):
            diff = H[u][v].get("total_difference", 0)
            center_shared_edge[(u, v)] = f"Total Shared: {w}\nTotal Difference: {diff}"
        else:
            other_edges[(u, v)] = f"{w}"
    
    nx.draw_networkx_edge_labels(H, pos, edge_labels=other_edges, font_color="red", font_size=14)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=shared_edges, font_color="green", font_size=14)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=center_shared_edge, font_color="navy", font_size=16, font_weight="bold")
    
    plt.axis("off")
    
    legend_handles = []
    cat_list = ["Supreme_Case", "Appeals_Case", "Statute", "Topic_ID"]
    for cat in cat_list:
        legend_handles.append(
            mlines.Line2D(
                [], [], marker='o', color='w',
                markerfacecolor=get_color_for_category(cat),
                markersize=12, label=cat.replace("_", " ")
            )
        )
    legend_handles.append(
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor="#5A2BB2", markersize=12, label="BOW estoppel")
    )
    legend_handles.append(
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor="red", markersize=12, label="KW estoppel")
    )
    plt.legend(
        handles=legend_handles,
        title='Node Types',
        loc="upper center",
        fontsize=15,
        title_fontsize=15,
        bbox_to_anchor=(.5, 1.3),

        borderaxespad=1.5,
        ncol=3
    )
    
    plt.tight_layout()
    plt.savefig("estoppel_graph.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()