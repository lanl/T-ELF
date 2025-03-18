import uuid
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Graph():

    RESERVED_NAMES = {  # reserved member variable names that cannot be used by kwargs for graph attribute(s)
        "nodes", "edges", "stats", "num_edges", 
        "num_nodes", "uid", "stat_funcs",
    }

    def __init__(self, G=None, **kwargs):
        if G is not None:
            self.G = G
        else:
            self.G = nx.DiGraph()
        
        self.stats = dict()
        self.num_edges = len(self.G.edges)
        self.num_nodes = len(self.G.nodes)
        self.uid = str(uuid.uuid4())

        # create a dictionary of supported stats to their callable functions
        prefix = 'get_'
        spacing = len(prefix)
        self.stat_funcs = {
            name[spacing:]: getattr(self, name) for name in dir(self) if 
            name.startswith(prefix) and name != 'get_stat' and callable(getattr(self, name))
        }

        # set any custom (global) attributes for the graph
        for k in kwargs:
            if k in self.RESERVED_NAMES:
                raise ValueError(f"'{k}' is reserved by the {self.__class__.__name__} class and cannot be used as an attribute name.")
            setattr(self, k, kwargs[k])


    def __len__(self):
        return self.num_nodes


    def add_node(self, node, **kwargs):
        """
        Add a single node to the graph object

        Parameters
        ----------
        node: str, numeric
            A unique identifier for the node in the graph
        kwargs: dict
            Arbitrary keyword arguments that will be stored as attributes of the node. The graph object 
            is allowed to contain duplicate attributes as long as node ids are unique.

        Returns
        -------
        None

        Example
        -------
        >>> g.add_node(0, name='Node name', affiliation='Affiliation Name')
        """
        if not isinstance(node, str):
            node = str(node)
        if self.G.has_node(node):
            raise ValueError(f'Node "{node}" already exists!')

        self.G.add_node(node, **kwargs)
        self.num_nodes += 1


    def add_edge(self, node1, node2, weight=None):
        """
        Creates and edge from node1 to node2 in the graph. 

        Parameters
        ----------
        node1: str, numeric
            A unique identifier for the first node in the graph
        node2: str, numeric
            A unique identifier for the second node in the graph
        weight: numeric; optional
            The weight for the connecting edge

        Returns
        -------
        None
        """
        node1, node2 = str(node1), str(node2)
        if not self.G.has_node(node1):
            raise ValueError(f'Could not find node "{node1}" in graph.')
        if not self.G.has_node(node2):
            raise ValueError(f'Could not find node "{node2}" in graph.')
        if self.G.has_edge(node1, node2):
            raise ValueError(f'Edge from node "{node1}" to node "{node2}" already exists!')
        if weight is None:
            weight = 1

        self.G.add_edge(node1, node2, weight=float(weight))
        self.num_edges += 1


    def subgraph(self, nodes):
        """
        Create a copy of the of the current Graph object that only preserves the nodes provided
        to this function. If only a single node is given, it is treated as an 'ego' node and the 
        nodes that are reachable from it within 1 step are computed. Otherwise a simple subgraph
        of the list of nodes is taken.

        Parameters
        ----------
        nodes: str, list
            List of node ids to preserve in the subgraph. Nodes in nodes that do not exist in the 
            graph will be ignored. If no nodes can be preserved, the funciton will return None.

        Returns
        -------
        new_graph: Graph, NoneType
            The resulting subgraph
        """
        kwargs = {k: v for k, v in self.__dict__.items() if k not in 
            self.RESERVED_NAMES and not k.startswith('_')}
        if isinstance(nodes, list):
            nodes = [str(x) for x in nodes]
            subgraph = nx.subgraph(self.G, nodes)
        elif isinstance(nodes, (int, str)):
            subgraph = nx.ego_graph(self.G, nodes)
        else:
            raise TypeError(f'Unexpected type for nodes: "{type(nodes)}"')
        return Graph(subgraph, **kwargs)

    
    def visualize(self, save_path=None, figsize=(8,8), seed=None, node_color=None, highlight_nodes=None, 
                        node_size=300, font_size=12, font_color='black', edge_width=1.0):
        """
        Visualize the current graph

        Parameters
        ----------
        save_path: str; optional
            If defined, the visualization will be saved at this path. Otherwise the figure will be 
            shown inline. Default=None
        iterations: int, optional
            Maximum number of iterations used by the layout algorithm. Default=50.
        seed: int, optional
            A random state that can be fixed to reproduce the same visualizations. If None, a random
            seed will be used. Default=None.

        Returns
        -------
        None
        """
        
        if highlight_nodes is not None and not isinstance(highlight_nodes, (list, tuple, dict)):
            raise ValueError('Invalid type passed for `higlight_nodes`!')
        
        components = nx.weakly_connected_components(self.G)  # find the connected components of the graph
        
        plt.figure(figsize=figsize)
        rightmost = 0
        for _, component in enumerate(components):  # ensure components are space by drawing them separately
            subgraph = self.G.subgraph(component)
            pos = nx.spring_layout(subgraph, seed=seed)

            # find the width of this component
            width = max(x for x, y in pos.values()) - min(x for x, y in pos.values())

            # set the default color
            if node_color is None:
                node_color = '#1f78b4'
            
            # establish the colormap
            if highlight_nodes is None:
                cmap = [node_color for node in subgraph]
            elif isinstance(highlight_nodes, (list, tuple)):
                cmap = ['red' if node in highlight_nodes else node_color for node in subgraph]
            else:
                cmap = [highlight_nodes.get(node, node_color) for node in subgraph]
            
            # draw the subgraph, shifting the x coordinates to the right of the last component drawn
            nx.draw_networkx(subgraph, pos={node: (x+rightmost, y) for node, (x, y) in pos.items()}, 
                             node_color=cmap, font_size=font_size, font_color=font_color, 
                             node_size=node_size, width=edge_width)
            rightmost += width + 1  # update the rightmost x-coordinate for the next component


        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


    def get_stat(self, stat, **kwargs):
        """
        Call a Graph stat function by the stat name. 

        Parameters
        ----------
        stat: str
            The name of the stat to be computed
        kwargs: dict; optional
            Any arguments required by the requested stat function

        Returns
        -------
        dict
            Dictionary with stat(s) and the graph uid as values
        """
        if stat not in self._stat_funcs:
            raise ValueError(f'"{stat}" is not a valid Graph stat\n'
                             f'Options are: {self.stat_funcs}')
        else: 
            func = self._stat_funcs[stat]
            return func(**kwargs)


    def get_page_rank(self, alpha=0.85, max_iter=1000):
        """
        Calculates the PageRank of the nodes in the graph. PageRank is a  ranking of the nodes 
        in the graph  based on the structure of the incoming links.

        Parameters
        ----------
        alpha: float; optional
            Damping parameter for PageRank, default=0.85
        max_iter: integer; optional
            Maximum number of iterations in power method eigenvalue solver, default=1000

        Returns
        -------
        dict
            Dictionary with page_rank and the graph uid as values
        """
        page_rank = nx.pagerank(self.G, alpha=alpha, max_iter=max_iter)
        self.stats["page_rank"] = page_rank
        return {"page_rank": page_rank, "uid": self.uid}


    def get_out_closeness(self):
        """
        Computes the outward closeness centrality for nodes in the graph. The closeness centrality of 
        a node u is the reciprocal of the average shortest outward path distance to u over all n-1 
        reachable nodes. 

        Returns
        -------
        dict
            Dictionary with in_closeness and the graph uid as values
        """
        out_closeness = nx.closeness_centrality(self.G.reverse())
        self.stats["out_closeness"] = out_closeness
        return {"out_closeness": out_closeness, "uid": self.uid}


    def get_in_closeness(self):
        """
        Computes the inward closeness centrality for nodes in the graph. The closeness centrality of 
        a node u is the reciprocal of the average shortest inward path distance to u over all n-1 
        reachable nodes. 

        Returns
        -------
        dict
            Dictionary with out_closeness and the graph uid as values
        """
        in_closeness = nx.closeness_centrality(self.G)
        self.stats["in_closeness"] = in_closeness
        return {"in_closeness": in_closeness, "uid": self.uid}


    def get_betweenness_centrality(self, k=None, normalized=True):
        """
        Computes the shortest-path betweenness centrality for nodes in the graph. Betweenness centrality 
        of a node u is the sum of the fraction of all-pairs shortest paths that pass through u.

        Parameters
        ----------
        k: int; optional
            Hyper-parameter for random sampling of nodes in betweenness_centrality. This parameter makes the 
            calculation O(kE) time complexity which is much faster than O(EV) alternative if k is not set. 
            Default=None
        normalized: bool; optional
            If True the betweenness values are normalized by 1/((n-1)(n-2)) for directed graphs where n is 
            the number of nodes in the graph. Default=True.

        Returns
        -------
        dict
            Dictionary with betweenness_centrality and the graph uid as values
        """
        betweenness_centrality = nx.betweenness_centrality(self.G, k=k, normalized=normalized)
        self.stats["betweenness_centrality"] = betweenness_centrality
        return {"betweenness_centrality": betweenness_centrality, "uid": self.uid}


    def get_out_degree(self):
        """
        Find the number of edges pointing out of each node in the graph.

        Returns
        -------
        dict
            Dictionary with out_degree and the graph uid as values
        """
        out_degree = {node: od for (node, od) in self.G.out_degree()}
        self.stats["out_degree"] = out_degree
        return {"out_degree": out_degree, "uid": self.uid}


    def get_in_degree(self):
        """
        Find the number of edges pointing into each node in the graph.

        Returns
        -------
        dict
            Dictionary with in_degree and the graph uid as values
        """
        in_degree = {node: od for (node, od) in self.G.in_degree()}
        self.stats["in_degree"] = in_degree
        return {"in_degree": in_degree, "uid": self.uid}


    def get_degree(self):
        """
        Find the number of edges connected to each node in the graph.

        Returns
        -------
        dict
            Dictionary with degree and the graph uid as values
        """
        in_degree = self.get_in_degree()['in_degree']
        out_degree = self.get_out_degree()['out_degree']
        degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in self.G.nodes}
        self.stats["degree"] = degree
        return {"degree": degree, "uid": self.uid}

    
    def get_in_weight(self):
        """
        Find the sum of weights of edges pointing into each node in the graph.

        Returns
        -------
        dict
            Dictionary with in_weight and the graph uid as values
        """
        in_weight = {node: sum(data.get('weight', 0) for _, _, data in self.G.in_edges(node, data=True)) 
                     for node in self.G.nodes()}
        self.stats["in_weight"] = in_weight
        return {"in_weight": in_weight, "uid": self.uid}

    
    def get_out_weight(self):
        """
        Find the sum of weights of edges pointing out of each node in the graph.

        Returns
        -------
        dict
            Dictionary with out_weight and the graph uid as values
        """
        out_weight = {node: sum(data.get('weight', 0) for _, _, data in self.G.out_edges(node, data=True)) 
                      for node in self.G.nodes()}
        self.stats["out_weight"] = out_weight
        return {"out_weight": out_weight, "uid": self.uid}

    
    def get_hubs_authorities(self, max_iter=100):
        """
        Computes the hubs and authorities values using the HITS algorithm. Authorities estimates the node 
        value based on the incoming links. Hubs estimates the node value based on outgoing links.

        Parameters
        ----------
        max_iter: integer; optional
            Maximum number of iterations in power method. Default=None.

        Returns
        -------
        dict
            Dictionary with hubs, authorities, and the graph uid as values
        """
        hubs, authorities = nx.hits(self.G, max_iter=max_iter)
        self.stats["hubs"] = hubs
        self.stats["authorities"] = authorities
        return {"hubs": hubs, "authorities": authorities, "uid": self.uid}


    def strongly_connected_components(self):
        """
        Get the strongly connected components of the graph as a list of new Graph objects where
        each member is one of the components. If the list is empty, no strongly connected 
        components exist

        Returns
        -------
        list
            A list of Graph objects, each of which is a strongly connected component of the
            original graph. The components are ordered by size with the largest component
            appearing first
        """
        scc = [list(x) for x in nx.strongly_connected_components(self.G)]
        scc = [self.subgraph(x) for x in scc]
        return sorted(scc, key=lambda x:  x.num_nodes, reverse=True)


    def weakly_connected_components(self):
        """
        Get the weakly connected components of the graph as a list of new Graph objects where
        each member is one of the components. If the list is empty, no weakly connected 
        components exist

        Returns
        -------
        list
            A list of Graph objects, each of which is a weakly connected component of the
            original graph. The components are ordered by size with the largest component
            appearing first
        """
        wcc = [list(x) for x in nx.weakly_connected_components(self.G)]
        wcc = [self.subgraph(x) for x in wcc]
        return sorted(wcc, key=lambda x:  x.num_nodes, reverse=True)


    def output_stats(self):

        # get all of the node attributes to setup data table
        node_attributes = set() 
        for node in self.G.nodes():
            attributes = self.G.nodes[node]
            for attribute in attributes:
                node_attributes.add(attribute)

        data = {'node_id': []}
        for attribute in node_attributes:
            data[attribute] = []
        for stat in self.stats:
            data[stat] = []


        for node in self.G.nodes():
            data['node_id'].append(node)
            attributes = self.G.nodes[node]
            for a in node_attributes:
                data[a].append(attributes.get(a))
            for s in self.stats:
                data[s].append(self.stats[s].get(node))

        return pd.DataFrame.from_dict(data)


    ### Getters


    @property
    def G(self):
        return self._G

    @property
    def stats(self):
        return self._stats

    @property
    def num_edges(self):
        return self._num_edges
    
    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def uid(self):
        return self._uid

    @property
    def stat_funcs(self):
        return sorted(list(self._stat_funcs.keys()), key=lambda x: len(x))


    ### Setters


    @G.setter
    def G(self, G):
        if not isinstance(G, nx.classes.digraph.DiGraph):
            raise ValueError('Expected G to be an nx.DiGraph object.')
        self._G = G

    @stats.setter
    def stats(self, stats):
        self._stats = stats

    @num_edges.setter
    def num_edges(self, num_edges):
        if not isinstance(num_edges, int):
            raise ValueError('num_edges must be an int')
        self._num_edges = num_edges

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        if not isinstance(num_nodes, int):
            raise ValueError('num_nodes must be an int')
        self._num_nodes = num_nodes

    @uid.setter
    def uid(self, uid):
        self._uid = uid

    @stat_funcs.setter
    def stat_funcs(self, stat_funcs):
        self._stat_funcs = stat_funcs
