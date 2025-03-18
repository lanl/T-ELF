from tqdm import tqdm
from joblib import Parallel, delayed

from .graph import Graph
from .utils import values_have_duplicates, keys_are_sequential, verify_matrix, verify_n_jobs, verify_stats



class Wolf:

    PARALLEL_BACKEND_OPTIONS = {
        'loky', 
        'multiprocessing', 
        'threading',
    }
    TARGET_STATS = {
        'in_weight',
        'out_weight',
        'degree',
        'in_degree',
        'out_degree',
        'page_rank', 
        'in_closeness',
        'out_closeness', 
        'hubs_authorities',
        'betweenness_centrality',
    }


    def __init__(self, n_jobs=-1, parallel_backend='loky', max_nbytes='100M', verbose=False):
        self.graphs = []
        self.node_ids = None
        self.attributes = None
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.max_nbytes = max_nbytes,
        self.verbose = verbose

        
    def create_graph(self, X, node_ids=None, attributes=None, target_nodes=None, alpha=None, use_weighted_value=True, **kwargs):
        """
        Create a graph from a matrix. The matrix can either be sparse or dense. 
        Note that the matrix cannot have entries on the diagonal. A diagonal entry would cause
        a loop in the graph. 

        Parameters
        ----------
        X: np.ndarray, scipy.sparse, pydata.sparse
            A dense or sparse 2-dimensional matrix. Non-zero entries in the matrix serve as an edge
            between the row, col entry.
        node_ids: dict, optional
            A map that can be used to replace the enumerated node identifiers with custom ids in the
            graph. 
        attributes: dict, optional
            A map that contains the attributes for the graph.
        target_nodes: list, NoneType
            List of node ids from which to create a subgraph. If the intersection of target_nodes and 
            node_ids is None, an empty graph will be created. If target_nodes is None, no subgraph
            will be taken. Default=None.
        alpha: int, NoneType
            A masking value that is used to remove noise from the matrix prior to creating the graph. 
            For an entry (i,j) into the matrix X, the entry is zeroed if it is less than alpha.
            Default=None. 
        use_weighted_value: bool
            Flag that determines if an edge in the graph is given a custom weight. If True, the weight
            for an edge between nodes i and j in the graph is the value at X(i,j). If false, every edge
            has a weight of 1. Default=True.
        kwargs: dict
            Arbitrary keyword arguments that will be stored as attributes of the graph. If the keyword is
            a reserved member variable of Graph, a ValueError will be thrown.

        Returns
        -------
        graph: Graph
            The created graph object. The output is also stored in the graphs attribute of the Wolf object.
        """
        g = Graph(**kwargs)
        
        # verify function input
        X = verify_matrix(X)
        if alpha is None:
            alpha = 0

        # update node ids and attributes if provided
        if node_ids is not None:
            self.node_ids = node_ids
        if attributes is not None:
            self.attributes = attributes
            
        X.data[X.data < alpha] = 0
        nnz_coords = X.nonzero()
        rows, cols = nnz_coords[0], nnz_coords[1]

        # add nodes
        for i in range(X.shape[0]):
            node = str(self.node_ids[0].get(i, i))
            if not g.G.has_node(node):
                node_attributes = self.attributes[0].get(node, {})
                g.add_node(node=node, **node_attributes)
        for i in range(X.shape[1]):
            node = str(self.node_ids[1].get(i, i))
            if not g.G.has_node(node):
                node_attributes = self.attributes[1].get(node, {})
                g.add_node(node=node, **node_attributes)

        # add edges
        data = X.data
        for i in range(X.nnz):
            n1 = str(self.node_ids[0].get(rows[i], rows[i]))
            n2 = str(self.node_ids[1].get(cols[i], cols[i]))
            weight = data[i] if use_weighted_value else None
            g.add_edge(n1, n2, weight)

        # get subgraph
        if target_nodes is not None:
            g = g.subgraph(target_nodes)
        return g

    
    @staticmethod
    def _apply_stats(g, stats):
        if isinstance(stats, dict):
            for s in stats:
                kwargs = stats[s]
                g.get_stat(s, **kwargs)
        else:
            for s in stats:
                g.get_stat(s)
        return g.output_stats()
    
    
    @staticmethod
    def _get_ranks_helper(X, stats, node_ids, attributes, target_nodes, alpha, use_weighted_value, slice_id, **kwargs):
        g = Wolf.create_graph(X = X, 
                              node_ids = node_ids,
                              attributes = attributes,
                              target_nodes = target_nodes,
                              alpha = alpha, 
                              use_weighted_value = use_weighted_value,
                              **kwargs)
        return (slice_id, Wolf._apply_stats(g, stats))
    
    
    def get_ranks(self, X, stats=None, slice_ids=None, node_ids=None, attributes=None, 
                  alpha=None, use_weighted_value=True, n_jobs=-1):
        """
        Compute graph statistics for a tensor.
        
        For each slice of a tensor, create a graph and evaluate said graph.

        Parameters
        ----------
        X: np.ndarray, pydata.sparse 
            A dense or sparse 3-dimensional tensor
        stats: list, dict
            The graph stats that should be calculated for the given tensor. If a list, each item in the 
            list should be a valid Graph stat function name. The arguments for this function will be kept
            as default. If a dict, the keys should be the function names and the values should be dicts
            of keyword arguments that should be passed to the respective stat function. If None, all
            stats will be computed using the default arguments. Default=None.
        slice_ids: dict, NoneType
            A map of labels for the third dimension of the tensor. The keys should be consecutive integers 
            starting with 0 and counting up to o for a tensor X of shape m by n by o. If slice_ids is None, 
            the enumeration of the slices will be used as the id.
            >>> slice_ids = {0: 2020, 1: 2021, 2: 2022, 3: 2023}
        node_ids: dict, [dict, dict], NoneType
            A map that can be used to replace the enumerated node identifiers with custom ids in the
            graph. The input is expected to be a dictionary, a list or tuple of two dictionaries, or None.
            If the input is a dict, it is assumed that both matrix dimensions share the same ids. If the
            input is a list/tuple the first dict is used for the first dimension and the second for the
            second. If the input is None, the enumerated ids are not replaced. Note that the keys of the 
            value should be sequential integers representing the original enumerated ids and the values
            should be the replacement ids. This argument is intended for use cases where the entries in 
            the matrix have unique identifiers that are more specific than the enumerated index values of 
            the matrix. For example, in the case of an documents x documents matrix, the enumerated ids can be 
            replaced with document ids. Default=None.
        attributes: dict, [dict, dict], NoneType
            A map that contains the attributes for the graph. The input is expected to be a dictionary, 
            a list or tuple of two dictionaries, or None. If the input is a dict, it is assumed that both 
            matrix dimensions share the same attributes. If the input is a list/tuple the first dict is 
            used for the first dimension and the second for the second. The key in this map is the id of 
            the node that will house the particular attribute(s). The values are dicts where the key is 
            the name of the attribute and the value is the attribute value. The node id for this argument 
            should match the format used by node_ids. If no attributes are provided, no attributes will 
            be set on the graph. Default=None.
        alpha: int, NoneType
            A masking value that is used to remove noise from the matrix prior to creating the graph. 
            For an entry (i,j) into the matrix X, the entry is zeroed if it is less than alpha.
            Default=None. 
        use_weighted_value: bool
            Flag that determines if an edge in the graph is given a custom weight. If True, the weight
            for an edge between nodes i and j in the graph is the value at X(i,j). If false, every edge
            has a weight of 1. Default=True.

        Returns
        -------
        dict:
            A dictionary where the keys are slices (either enumerated ids or more specific ids provided 
            by the dimension map `slice_ids`). The values are pandas DataFrames that contain graph 
            statistics for the corresponding slice of the tensor. 
        """
        if stats is None:
            stats = list(Wolf.TARGET_STATS)
        if slice_ids is None:
            slice_ids = {}
        
        # verify the function input
        verify_stats(stats)
        self.node_ids = node_ids
        self.attributes = attributes
        self.n_jobs = n_jobs

        # form the graphs in parallel
        slices = [X[:, :, i].copy() for i in range(X.shape[2])]
    
        if self.n_jobs == 1:
            output = {}
            for i,s in tqdm(enumerate(slices), total=len(slices), disable=not self.verbose):
                slice_id = slice_ids.get(i,i)
                slice_id, results = self._get_ranks_helper(X=s, 
                                                           stats=stats,
                                                           node_ids=self.node_ids, 
                                                           attributes=self.attributes, 
                                                           target_nodes=None, 
                                                           alpha=alpha, 
                                                           use_weighted_value=use_weighted_value,
                                                           slice_id=slice_id)
                output[slice_id] = results
            return output
        else:
            results = Parallel(n_jobs=self.n_jobs, max_nbytes=self.max_nbytes, 
                               backend=self.parallel_backend, verbose=self.verbose)(
                          delayed(self._get_ranks_helper)(
                              X=s, 
                              stats=stats,
                              node_ids=self.node_ids, 
                              attributes=self.attributes, 
                              target_nodes=None, 
                              alpha=alpha, 
                              use_weighted_value=use_weighted_value,
                              slice_id = slice_ids.get(i,i)
                          )
                          for i, s in enumerate(slices)
            )
            return dict(results)


    def get_community_ranks(self, X, A, stats=None, node_ids=None, attributes=None, top_n=10, alpha=None):
        
        # compute the year groups
        third_dim_groups = []
        for item in list(range(0, X.shape[-1])):
            for group in list(range(0, A.shape[-1])):
                third_dim_groups.append((item, group))

        # verify the function stats
        if stats is None:
            stats = list(Wolf.TARGET_STATS)
        verify_stats(stats)
        self.node_ids = node_ids
        self.attributes = attributes
        
        target_nodes = []
        assert isinstance(top_n, int), "If using A, pass integer top_n!"
        for yg in third_dim_groups:
            target_indices = list(map(str, A[:, yg[1]].argsort()[-top_n:][::-1]))
            if node_ids is None:
                target_nodes.append(target_indices)
            else:
                target_nodes.append([node_ids[int(i)] for i in target_indices])

        if self.n_jobs == 1:
            output = {}
            for i, yg in tqdm(enumerate(third_dim_groups), total=len(third_dim_groups), disable=not self.verbose):
                slice_id = yg
                slice_id, results = self._get_ranks_helper(X=X[:, :, yg[0]],
                                                           stats=stats,
                                                           node_ids=self.node_ids, 
                                                           attributes=self.attributes, 
                                                           target_nodes=target_nodes[i],
                                                           alpha=alpha, 
                                                           use_weighted_value=True,
                                                           slice_id=slice_id,
                                                           year=yg[0], 
                                                           group=yg[1])
                output[slice_id] = results
            return output
        else:
            results = Parallel(n_jobs=self.n_jobs, max_nbytes=self.max_nbytes, 
                               backend=self.parallel_backend, verbose=self.verbose)(
                          delayed(self._get_ranks_helper)(
                              X=X[:, :, yg[0]],
                              stats=stats,
                              node_ids=self.node_ids, 
                              attributes=self.attributes, 
                              target_nodes=target_nodes[i],
                              alpha=alpha, 
                              use_weighted_value=True,
                              slice_id=yg,
                              year=yg[0], 
                              group=yg[1])
                          for i, yg in enumerate(third_dim_groups)
            )
            return dict(results)
        
        
    ### GETTERS / SETTERS


    @property
    def node_ids(self):
        return self._node_ids

    @property
    def attributes(self):
        return self._attributes

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def parallel_backend(self):
        return self._parallel_backend
    
    @property
    def verbose(self):
        return self._verbose

    
    ### Setters


    @node_ids.setter
    def node_ids(self, node_ids):
        if isinstance(node_ids, dict):  # single dictionary, assuming that both dimensions are same
            self._node_ids = {0: node_ids, 1: node_ids}
        elif isinstance(node_ids, (list, tuple)) and len(node_ids) == 2 and all(isinstance(item, dict) for item in node_ids):
            self._node_ids = {0: node_ids[0], 1: node_ids[1]}
        elif node_ids is None:
            self._node_ids = {0: {}, 1: {}}
        else:
            raise TypeError('node_ids must be a dictionary, a list or tuple of two dictionaries, or None')

        for dim in self.node_ids:
            if not keys_are_sequential(self.node_ids[dim]):
                raise ValueError('Keys of node_ids must be sequntial integers representing the index of matrix')
            if values_have_duplicates(self.node_ids[dim]):
                raise ValueError('Values of node_ids cannot contain any duplicates')

    @attributes.setter
    def attributes(self, attributes):
        if isinstance(attributes, dict):  # single dictionary, assuming that both dimensions are same
            self._attributes = {0: attributes, 1: attributes}
        elif isinstance(attributes, (list, tuple)) and \
             len(attributes) == 2 and \
             all(isinstance(item, dict) for item in attributes):
            self._attributes = {0: attributes[0], 1: attributes[1]}
        elif attributes is None:
            self._attributes = {0: {}, 1: {}}
        else:
            raise TypeError('attributes must be a dictionary, a list or tuple of two dictionaries, or None')

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = verify_n_jobs(n_jobs)
        
    @parallel_backend.setter
    def parallel_backend(self, parallel_backend):
        if not isinstance(parallel_backend, str):
            raise TypeError('parallel_backend must be an str!')
        if parallel_backend not in self.PARALLEL_BACKEND_OPTIONS:
            raise ValueError(f'{parallel_backend} is not a valid parallel_backend option!')
        self._parallel_backend = parallel_backend

    @verbose.setter
    def verbose(self, verbose):
        if isinstance(verbose, bool):
            self._verbose = int(verbose)  # convert False to 0, True to 1
        elif isinstance(verbose, int):
            if verbose < 0:
                raise ValueError("Integer values for verbose must be non-negative!")
            self._verbose = verbose
        else:
            raise TypeError("verbose should be of type bool or int!")