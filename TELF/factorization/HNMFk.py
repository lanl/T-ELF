from .NMFk import NMFk
from pathlib import Path
import numpy as np
import uuid
import os


class Node():
    def __init__(self,
                 node_num: int,
                 depth: int,
                 parent_topic: int,
                 W: np.ndarray, H: np.ndarray, k: int,
                 parent_node, parent_node_name: str,
                 child_nodes: list, child_node_names: list,
                 original_indices: np.ndarray,
                 num_samples:int,
                 leaf:bool,
                 user_node_data:dict
                 ):
        self.node_num = node_num
        self.name = "node"+str(node_num)
        self.node_id = str(uuid.uuid1())
        self.depth = depth
        self.W = W
        self.H = H
        self.k = k
        self.parent_topic = parent_topic
        self.parent_node = parent_node
        self.parent_node_name = parent_node_name
        self.child_nodes = child_nodes
        self.child_node_names = child_node_names
        self.original_indices = original_indices
        self.num_samples = num_samples
        self.leaf = leaf
        self.user_node_data = user_node_data


class HNMFk():

    def __init__(self,
                 nmfk_params=[{}],
                 cluster_on="H",
                 depth=1,
                 sample_thresh=-1,
                 Ks_deep_min=1,
                 Ks_deep_max=None,
                 Ks_deep_step=1,
                 K2=False,
                 experiment_name="HNMFk_Output",
                 generate_X_callback=None
                 ):
        """
        HNMFk is a Hierarchical Non-negative Matrix Factorization module with the capability to do automatic model determination.

        Parameters
        ----------
        nmfk_params : list of dicts, optional
            We can specify NMFk parameters for each depth, or use same for all depth.\n
            If there is single items in ``nmfk_params``, HMMFk will use the same NMFk parameters for all depths.\n
            When using for each depth, append to the list. For example, [nmfk_params0, nmfk_params1, nmfk_params2] for depth of 2
            The default is ``[{}]``, which defaults to NMFk with defaults with required ``params["collect_output"] = False``, ``params["save_output"] = True``, and ``params["predict_k"] = True`` when ``K2=False``.
        cluster_on : str, optional
            Where to perform clustering, can be W or H. Ff W, row of X should be samples, and if H, columns of X should be samples.
            The default is "H".
        depth : int, optional
            How deep to go in each topic after root node. if -1, it goes until samples cannot be seperated further. The default is 1.
        sample_thresh : int, optional
            Stopping criteria for num of samples in the cluster. 
            When -1, this criteria is not used.
            The default is -1.
        Ks_deep_min : int, optional
            After first nmfk, when selecting Ks search range, minimum k to start. The default is 1.
        Ks_deep_max : int, optinal
            After first nmfk, when selecting Ks search range, maximum k to try.\n
            When None, maximum k will be same as k selected for parent node.\n
            The default is None.
        Ks_deep_step : int, optional
            After first nmfk, when selecting Ks search range, k step size. The default is 1.
        K2 : bool, optional
            If K2=True, decomposition is done only for k=2 instead of finding and predicting the number of stable latent features. The default is False.
        experiment_name : str, optional
            Where to save the results.
        generate_X_callback : object, optional
            This can be used to re-generate the data matrix X before each NMFk operation. When not used, slice of original X is taken, which is equal to serial decomposition.\n
            ``generate_X_callback`` object should be a class with ``def __call__(original_indices)`` defined so that ``new_X, save_at_node=generate_X_callback(original_indices)`` can be done.\n
            ``original_indices`` hyper-parameter is the indices of samples (columns of original X when clustering on H).\n
            Here ``save_at_node`` is a dictionary that can be used to save additional information in each node's ``user_node_data`` variable. 
            The default is None.
        Returns
        -------
        None.

        """

        self.sample_thresh = sample_thresh
        self.depth = depth
        self.cluster_on = cluster_on
        self.Ks_deep_min = Ks_deep_min
        self.Ks_deep_max = Ks_deep_max
        self.Ks_deep_step = Ks_deep_step
        self.K2 = K2
        self.experiment_name = experiment_name
        self.generate_X_callback = generate_X_callback

        organized_nmfk_params = []
        for params in nmfk_params:
            organized_nmfk_params.append(self._organize_nmfk_params(params))
        self.nmfk_params = organized_nmfk_params

        self.target_clusters = []
        self.target_Ks = []
        self.curr_original_indices = []
        self.X = None
        self.num_features = 0
        self.node_count = 0
        self._all_nodes = []

        self.root = Node(
            node_num=self.node_count,
            depth=0,
            W=None, H=None,
            k=-1,
            parent_topic=None,
            parent_node=None,
            parent_node_name="None",
            child_nodes=[],
            child_node_names=[],
            original_indices=None,
            num_samples=0,
            leaf=False,
            user_node_data={}
        )
        self.iterator = self.root

        assert self.cluster_on in ["W", "H"], "Unknown clustering method!"

        self.experiment_save_path = os.path.join(self.experiment_name)
        try:
            if not Path(self.experiment_save_path).is_dir():
                Path(self.experiment_save_path).mkdir(parents=True)
        except Exception as e:
            print(e)

    def fit(self, X, Ks):
        """
        Factorize the input matrix ``X`` for the each given K value in ``Ks``.

        Parameters
        ----------
        X : ``np.ndarray`` or ``scipy.sparse._csr.csr_matrix`` matrix
            Input matrix to be factorized.
        Ks : list
            List of K values to factorize the input matrix.\n
            **Example:** ``Ks=range(1, 10, 1)``.
        
        Returns
        -------
        None
        """

        self.X = X

        if self.cluster_on == "W":
            self.root.original_indices = np.arange(0, X.shape[0], 1)
            self.root.num_samples = X.shape[0]
            self.num_features = X.shape[1]
            self.root.leaf = False

        elif self.cluster_on == "H":
            self.root.original_indices = np.arange(0, X.shape[1], 1)
            self.num_features = X.shape[0]
            self.root.num_samples = X.shape[1]
            self.root.leaf = False

        # begin
        self._hierarchical_nmfk(self.root, Ks)

    def _hierarchical_nmfk(self, node, Ks):

        # check if only one sample is in the node
        if node.num_samples == 1:
            node.leaf = True
            return

        # check if num samples in node below threshold
        if (self.sample_thresh > 0 and (node.num_samples <= self.sample_thresh)):
            node.leaf = True
            return
        
        # where to save current depth
        try:
            depth_save_path = os.path.join(self.experiment_name, "depth_"+str(node.depth))
            try:
                if not Path(depth_save_path).is_dir():
                    Path(depth_save_path).mkdir(parents=True)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)

        # obtain the current X using the original X
        if self.generate_X_callback is None or node.depth == 0:
            save_at_node = {}
            if self.cluster_on == "W":
                curr_X = self.X[node.original_indices]

            elif self.cluster_on == "H":
                curr_X = self.X[:, node.original_indices]
        # obtain the current X using the callback function
        else:
            curr_X, save_at_node = self.generate_X_callback(node.original_indices)

        # prepare directory to save the results
        if node.depth >= len(self.nmfk_params):
            select_params = -1
        else:
            select_params = node.depth 
        curr_nmfk_params = self.nmfk_params[select_params % len(self.nmfk_params)]
        curr_nmfk_params["save_path"] = depth_save_path

        # apply nmfk
        model = NMFk(**curr_nmfk_params)
        parent = node.parent_node
        
        if parent is None:
            folder_name = node.name
        else:
            folder_name = node.name+"-parent-"+str(node.parent_node.name)

        results = model.fit(curr_X, Ks, name=folder_name)
        
        # Check if decomposition was not possible
        if results is None:
            node.leaf = True
            return

        if self.K2:
            factors_data = np.load(f'{model.save_path_full}/WH_k=2.npz')
            node.W = factors_data["W"]
            node.H = factors_data["H"]
            node.k = 2
        
        else:
            predict_k = results["k_predict"]
            factors_data = np.load(f'{model.save_path_full}/WH_k={predict_k}.npz')
            node.W = factors_data["W"]
            node.H = factors_data["H"]
            node.k = predict_k
            

        # obtain the clusters
        if self.cluster_on == "W":
            cluster_labels = np.argmax(node.W, axis=1)
            clusters = np.arange(0, node.W.shape[1], 1)

        elif self.cluster_on == "H":
            cluster_labels = np.argmax(node.H, axis=0)
            clusters = np.arange(0, node.H.shape[0], 1)

        # obtain the unique number of clusters that samples falls to
        n_clusters = len(set(cluster_labels))

        # leaf node or single cluster or all samples in same cluster
        if ((node.depth >= self.depth) and self.depth > 0) or node.k == 1 or n_clusters == 1:
            node.leaf = True
            return

        # go through each topic
        for c in clusters:

            # current cluster samples
            cluster_c_indices = np.argwhere(cluster_labels == c).flatten()

            # empty cluster
            if len(cluster_c_indices) == 0:
                continue

            # create a new child node for the cluster
            self.node_count += 1
            child_node = Node(
                node_num=self.node_count,
                depth=node.depth + 1,
                W=None, H=None,
                k=-1,
                parent_topic=c,
                parent_node=node,
                parent_node_name=node.name,
                child_nodes=[],
                child_node_names=[],
                original_indices=node.original_indices[cluster_c_indices],
                num_samples=len(cluster_c_indices),
                leaf=False,
                user_node_data=save_at_node
            )
            node.child_nodes.append(child_node)
            node.child_node_names.append(child_node.name)

            # prepare to go to the next depth
            if not self.K2:
                if self.Ks_deep_max is None:
                    k_max = node.k + 1
                else:
                    k_max = self.Ks_deep_max + 1

                new_max_K = min(k_max, min(len(child_node.original_indices), self.num_features))
                new_Ks = range(self.Ks_deep_min, new_max_K, self.Ks_deep_step)
            else:
                new_Ks = [2]
                
            self._hierarchical_nmfk(child_node, new_Ks)

        return

    def traverse_nodes(self):
        """
        Graph iterator. Returns all nodes in list format.

        Returns
        -------
        data : list
            List of all nodes where each node is a dictionary.

        """
        self._all_nodes = []
        self._get_traversal(self.root)
        return_data = self._all_nodes.copy()
        self._all_nodes = []

        return return_data

    def _get_traversal(self, node):

        for nn in node.child_nodes:
            self._get_traversal(nn)

        data = vars(node).copy()
        del data["child_nodes"]
        del data["parent_node"]

        self._all_nodes.append(data)

    def go_to_root(self):
        """
        Graph iterator. Goes to root node.

        Returns
        -------
        data : dict
            Dictionary format of node.

        """
        self.iterator = self.root
        data = vars(self.iterator).copy()
        del data["child_nodes"]
        del data["parent_node"]
        return data

    def get_node(self):
        """
        Graph iterator. Returns the current node.

        Returns
        -------
        data : dict
            Dictionary format of node.

        """
        data = vars(self.iterator).copy()
        del data["child_nodes"]
        del data["parent_node"]
        return data

    def go_to_parent(self):
        """
        Graph iterator. Goes to the parent of current node.

        Returns
        -------
        data : dict
            Dictionary format of node.
        """
        if self.iterator is not None:
            self.iterator = self.iterator.parent_node
            data = vars(self.iterator).copy()
            del data["child_nodes"]
            del data["parent_node"]
            return data
        else:
            print("At the root! There is no parents.")

    def go_to_children(self, idx: int):
        """
        Graph iterator. Goes to the child node specified by index.

        Parameters
        ----------
        idx : int
            Child index.

        Returns
        -------
        data : dict
            Dictionary format of node.
        """

        try:
            self.iterator = self.iterator.child_nodes[idx]
            data = vars(self.iterator).copy()
            del data["child_nodes"]
            del data["parent_node"]
            return data

        except:
            print("Children at index "+str(idx)+" from the current does not exist!")

    def _organize_nmfk_params(self, params):

        #
        # Defaults
        #
        if "n_perturbs" not in params:
            params["n_perturbs"] = 20

        if "n_iters" not in params:
            params["n_iters"] = 250

        if "epsilon" not in params:
            params["epsilon"] = 0.005

        if "n_jobs" not in params:
            params["n_jobs"] = -1

        if "use_gpu" not in params:
            params["use_gpu"] = False

        if "init" not in params:
            params["init"] = "nnsvd"

        if "nmf_method" not in params:
            params["nmf_method"] = "nmf_fro_mu"

        if "sill_thresh" not in params:
            params["sill_thresh"] = 0.8

        #
        # Required
        #
        params["collect_output"] = False
        params["save_output"] = True
        params["n_nodes"] = 1
        
        if not self.K2:
            params["predict_k"] = True


        return params
