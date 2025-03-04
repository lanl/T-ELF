"""
job-schedule - 200
job-complete - 300
signal-exit  - 400
"""

from .NMFk import NMFk
from pathlib import Path
import numpy as np
import uuid
import os
import time
import pickle
import warnings

from .utilities.hpc_comm_helpers import signal_workers_exit, worker_check_exit_status, get_next_job_at_worker, collect_results_from_workers, send_job_to_worker_nodes

try:
    from mpi4py import MPI
except:
    MPI = None


class OnlineNode():
    def __init__(self, 
                 node_path:str,
                 node_name:str,
                 parent_node:object,
                 child_nodes:list,
                 ) -> None:
        self.node_path = node_path
        self.node_name = node_name
        self.parent_node = parent_node
        self.child_nodes = child_nodes
        self.original_child_nodes = child_nodes
    
    def __call__(self):
        return pickle.load(open(self.node_path, "rb"))


class Node():
    def __init__(self,
                 node_name: str,
                 depth: int,
                 parent_topic: int,
                 parent_node_k: int,
                 W: np.ndarray, H: np.ndarray, k: int,
                 parent_node_name: str,
                 child_node_names: list,
                 original_indices: np.ndarray,
                 num_samples:int,
                 leaf:bool,
                 user_node_data:dict,
                 cluster_indices_in_parent:list,
                 node_save_path:str,
                 parent_node_save_path:str,
                 parent_node_factors_path:str,
                 exception:bool,
                 signature:np.array,
                 probabilities:np.array,
                 centroids:np.array,
                 factors_path:str,
                 ):
        self.node_name = node_name
        self.depth = depth
        self.W = W
        self.H = H
        self.k = k
        self.parent_topic = parent_topic
        self.parent_node_k = parent_node_k
        self.parent_node_name = parent_node_name
        self.child_node_names = child_node_names
        self.original_child_node_names = child_node_names
        self.original_indices = original_indices
        self.num_samples = num_samples
        self.leaf = leaf
        self.user_node_data = user_node_data
        self.cluster_indices_in_parent = cluster_indices_in_parent
        self.node_save_path=node_save_path
        self.parent_node_factors_path=parent_node_factors_path
        self.parent_node_save_path=parent_node_save_path
        self.exception = exception
        self.signature = signature
        self.probabilities = probabilities
        self.centroids = centroids
        self.factors_path = factors_path


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
                 generate_X_callback=None,
                 n_nodes=1,
                 verbose=True,
                 comm_buff_size=10000000,
                 random_identifiers=False,
                 root_node_name = "Root"
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
        n_nodes : int, optional
            Number of HPC nodes. The default is 1.
        verbose : bool, optional
            If True, it prints progress. The default is True.
        random_identifiers : bool, optional
            If True, model will use randomly generated strings as the identifiers of the nodes. Otherwise, it will use the k for ancestry naming convention. 
        root_node_name : str, optional
            Naming convention to be used when saving the root name. Default is "Root".
        Returns
        -------
        None.

        """

        # user defined settings
        self.sample_thresh = sample_thresh
        self.depth = depth
        self.cluster_on = cluster_on
        self.Ks_deep_min = Ks_deep_min
        self.Ks_deep_max = Ks_deep_max
        self.Ks_deep_step = Ks_deep_step
        self.K2 = K2
        self.experiment_name = experiment_name
        self.generate_X_callback = generate_X_callback
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.comm_buff_size = comm_buff_size
        self.random_identifiers = random_identifiers
        self.root_node_name = root_node_name

        organized_nmfk_params = []
        for params in nmfk_params:
            organized_nmfk_params.append(self._organize_nmfk_params(params))
        self.nmfk_params = organized_nmfk_params

        # object variables
        self.X = None
        self.total_exec_seconds = 0
        self.num_features = 0
        self._all_nodes = []
        self.iterator = None
        self.root = None
        self.root_name = ""
        self.node_save_paths = {}

        # path to save 
        self.experiment_save_path = os.path.join(self.experiment_name)
        try:
            if not Path(self.experiment_save_path).is_dir():
                Path(self.experiment_save_path).mkdir(parents=True)
        except Exception as e:
            print(e)

        # HPC job management variables
        self.target_jobs = {}
        self.node_status = {}

        assert self.cluster_on in ["W", "H"], "Unknown clustering method!"
        assert (self.n_nodes > 1 and MPI is not None) or (self.n_nodes ==
                                                          1), "n_nodes was greater than 1 but MPI is not installed!"
        
    def load_model(self):
        """
        Loads existing model from checkpoint file located at self.experiment_name path.\n 
        This checkpoint file exist if fit(save_checkpoint=True) when the model was run.\n
        Use this function to leverage the graph iterator for an existing model.

        Returns
        -------
        None

        """
        checkpoint_file = self._load_checkpoint_file()
        if len(checkpoint_file) > 0:
            backup_name = self.experiment_name
            backup_path = self.experiment_save_path
            self._load_checkpoint(checkpoint_file)
            self.experiment_name = backup_name
            self.experiment_save_path = backup_path
        else:
            raise Exception(f"Model checkpoint file is not found at: {os.path.join(self.experiment_save_path, 'checkpoint.p')}")
        
        # correct node save paths
        new_node_save_paths = {}
        for key, path in self.node_save_paths.items():
            new_node_save_paths[key] = os.path.join(
                self.experiment_name, 
                *path.split(os.sep)[1:])
        self.node_save_paths = new_node_save_paths.copy()
        del new_node_save_paths

        # prepare online iterator
        self.root = OnlineNode(
            node_path=self.node_save_paths[self.root_name],
            node_name = self.root_name,
            parent_node=None,
            child_nodes=[]
        )
        self.iterator = self.root
        self._prepare_iterator(self.root)


    def fit(self, X, Ks, from_checkpoint=False, save_checkpoint=True):
        """
        Factorize the input matrix ``X`` for the each given K value in ``Ks``.

        Parameters
        ----------
        X : ``np.ndarray`` or ``scipy.sparse._csr.csr_matrix`` matrix
            Input matrix to be factorized.
        Ks : list
            List of K values to factorize the input matrix.\n
            **Example:** ``Ks=range(1, 10, 1)``.
        from_checkpoint : bool, optional
            If True, it continues the process from the checkpoint. The default is False.
        save_checkpoint : bool, optional
            If True, it saves checkpoints. The default is True.
        
        Returns
        -------
        None
        """
        #
        # Job scheduling setup
        #

        # multi-node processing
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        # single node processing
        else:
            comm = None
            rank = 0

        #
        # Checkpointing and job setup
        #   
        if from_checkpoint:
            checkpoint_file = self._load_checkpoint_file()
        else:
            checkpoint_file = {}
        
        if from_checkpoint and len(checkpoint_file) > 0:
            if self.verbose:
                print("Continuing from checkpoint...")

            if rank == 0:
                self._load_checkpoint(checkpoint_file)

        
        # setting up for a new job
        elif not from_checkpoint or len(checkpoint_file) == 0:
            
            if rank == 0:
                if self.cluster_on == "W":
                    original_indices = np.arange(0, X.shape[0], 1)
                elif self.cluster_on == "H":
                    original_indices = np.arange(0, X.shape[1], 1)
                    
                if self.random_identifiers:
                    self.root_name = str(uuid.uuid1())
                else:
                    self.root_name = self.root_node_name
                    
                self.target_jobs[self.root_name] = {
                    "parent_node_name":"None",
                    "node_name":self.root_name,
                    "Ks":Ks,
                    "original_indices":original_indices,
                    "cluster_indices_in_parent":[],
                    "depth":0,
                    "parent_topic":None,
                    "parent_node_save_path":None,
                    "parent_node_factors_path":None,
                    "parent_node_k":None,
                }
        
        # organize node status
        if self.n_nodes > 1:
            self.node_status = {}
            for ii in range(1, self.n_nodes, 1):
                self.node_status[ii] = {"free":True, "job":None}

        else:
             self.node_status = {}
             self.node_status[0] = {"free": True, "job":None}

        # save data matrix
        self.X = X 
        if self.cluster_on == "W":
            self.num_features = X.shape[1]
        elif self.cluster_on == "H":
            self.num_features = X.shape[0]
        else:
            raise Exception("Unknown clustering method!")

        # wait for everyone
        start_time = time.time()
        if self.n_nodes > 1:
            comm.Barrier()

        #
        # Run HNMFk
        # 
        while True:

            # check exit status 
            if len(self.target_jobs) == 0 and rank == 0 and all([info["free"] for _, info in self.node_status.items()]):
                if self.n_nodes > 1:
                    signal_workers_exit(comm, self.n_nodes)
                break
                
            #
            # worker nodes check exit status
            #
            if self.n_nodes > 1:
                worker_check_exit_status(rank, comm)

            #
            # send job to worker nodes
            #
            if rank == 0 and self.n_nodes > 1:
                send_job_to_worker_nodes(
                    comm, self.target_jobs, self.node_status
                )

            #
            # recieve jobs from rank 0 at worker nodes
            #
            elif rank != 0 and self.n_nodes > 1:
                job_data, job_flag = get_next_job_at_worker(
                    rank, comm, self.comm_buff_size
                )

            #
            # single node job schedule
            #
            else:
                available_jobs = list(self.target_jobs.keys())
                next_job = available_jobs.pop(0)
                job_data = self.target_jobs[next_job]
                job_flag = True


            # do the job
            if (rank != 0 and job_flag) or (self.n_nodes == 1 and rank == 0):
                
                # process the current node
                node_results = self._process_node(**job_data)

                # send worker node results to root
                if self.n_nodes > 1 and rank != 0:
                    req = comm.isend(node_results, dest=0, tag=int(f'300{rank}'))
                    req.wait()

            # collect results at root
            elif rank == 0 and self.n_nodes > 1:
                all_node_results = collect_results_from_workers(
                    rank, comm, self.n_nodes, 
                    self.node_status, self.comm_buff_size
                )
                if len(all_node_results) == 0:
                    continue
            
            # process results for job scheduling
            if rank == 0:
                if self.n_nodes > 1:
                    for node_results in all_node_results:
                        self._process_results(node_results, save_checkpoint=save_checkpoint)
                else:
                    self._process_results(node_results, save_checkpoint=save_checkpoint)

        # total execution time
        total_exec_seconds = time.time() - start_time
        self.total_exec_seconds = total_exec_seconds

        # prepare online iterator
        self.root = OnlineNode(
            node_path=self.node_save_paths[self.root_name],
            node_name = self.root_name,
            parent_node=None,
            child_nodes=[]
        )
        self.iterator = self.root
        self._prepare_iterator(self.root)

        # checkpointing at the final results
        if save_checkpoint:
            self._save_checkpoint()

        if self.verbose:
            print("Done")

        return {"time":total_exec_seconds}


    def _prepare_iterator(self, node):
        node_object = pickle.load(open(self.node_save_paths[node.node_name], "rb"))
        child_node_names = node_object.child_node_names

        for child_name in child_node_names:
            # check if node is not done its job
            if child_name in self.target_jobs:
                continue
            # if done, load
            child_node = OnlineNode(
                node_path=self.node_save_paths[child_name],
                node_name = child_name,
                parent_node=node,
                child_nodes=[])
            node.child_nodes.append(child_node)
            self._prepare_iterator(child_node)

        return
        
    def _process_node(self, Ks, 
                      depth, 
                      original_indices, cluster_indices_in_parent, 
                      node_name, parent_node_name, 
                      parent_topic, parent_node_save_path,
                      parent_node_factors_path, parent_node_k):
        # where to save current node
        try:
            node_save_path = os.path.join(self.experiment_name, "depth_"+str(depth), node_name)
            try:
                if not Path(node_save_path).is_dir():
                    Path(node_save_path).mkdir(parents=True)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)

        #
        # Create a node object
        #
        target_jobs = []
        try:
            parent_factors_data = np.load(parent_node_factors_path)
            signature = parent_factors_data["W"][:,int(parent_topic)]
            probabilities = (parent_factors_data["H"] / parent_factors_data["H"].sum(axis=0))[int(parent_topic)][cluster_indices_in_parent]
            centroids = (parent_factors_data["H"] / parent_factors_data["H"].sum(axis=0))[:,cluster_indices_in_parent]

        except Exception:
            signature = None
            probabilities = None
            centroids = None

        current_node = Node(
                node_name=node_name,
                node_save_path=os.path.join(str(node_save_path), f'node_{node_name}.p'),
                parent_node_save_path=parent_node_save_path,
                depth=depth,
                parent_topic=parent_topic,
                parent_node_name=parent_node_name,
                original_indices=original_indices,
                num_samples=len(original_indices),
                child_node_names=[],
                user_node_data={},
                leaf=False,
                W=None, H=None,
                k=None,
                factors_path=None,
                cluster_indices_in_parent=cluster_indices_in_parent,
                exception=False,
                parent_node_k=parent_node_k,
                parent_node_factors_path=parent_node_factors_path,
                signature=signature,
                probabilities=probabilities,
                centroids=centroids,
        )

        #
        # check if leaf node status based on number of samples
        #
        if (current_node.num_samples == 1):
            current_node.leaf = True
            current_node.exception = True
            pickle_path = os.path.join(str(node_save_path), f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}
        
        
        #
        # Sample threshold check for leaf node determination
        #
        if self.sample_thresh > 0 and (current_node.num_samples <= self.sample_thresh):
            current_node.leaf = True 
            pickle_path = os.path.join(f'{node_save_path}', f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}

        #
        # obtain the current X
        #
        if self.generate_X_callback is None or current_node.depth == 0:
            save_at_node = {}
            if self.cluster_on == "W":
                curr_X = self.X[current_node.original_indices]

            elif self.cluster_on == "H":
                curr_X = self.X[:, current_node.original_indices]

        else:
            curr_X, save_at_node = self.generate_X_callback(current_node.original_indices)
            current_node.user_node_data = save_at_node.copy()

        #
        # Based on number of features or samples, no seperation possible
        #
        if min(curr_X.shape) <= 1:
            current_node.leaf = True
            current_node.exception = True 
            pickle_path = os.path.join(f'{node_save_path}', f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}

        #
        # prepare the current nmfk parameters
        #
        if current_node.depth >= len(self.nmfk_params):
            select_params = -1
        else:
            select_params = current_node.depth 
        curr_nmfk_params = self.nmfk_params[select_params % len(self.nmfk_params)]
        curr_nmfk_params["save_path"] = node_save_path

        #
        # check for K range
        #
        Ks = self._adjust_curr_Ks(curr_X.shape, Ks)
        if len(Ks) == 0 or (len(Ks) == 1 and Ks[0] < 2):
            current_node.leaf = True
            current_node.exception = True 
            pickle_path = os.path.join(str(node_save_path), f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}

        #
        # apply nmfk
        #
        model = NMFk(**curr_nmfk_params)
        results = model.fit(curr_X, Ks, name=f'NMFk-{node_name}')
        
        #
        # Check if decomposition was not possible
        #
        if results is None:
            current_node.leaf = True
            current_node.exception = True
            pickle_path = os.path.join(str(node_save_path), f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}

        #
        # latent factors
        # 
        if self.K2:
            factors_path = os.path.join(f'{model.save_path_full}', "WH_k=2.npz")
            factors_data = np.load(factors_path)
            current_node.W = factors_data["W"]
            current_node.H = factors_data["H"]
            current_node.k = 2
            current_node.factors_path = factors_path
        
        else:
            predict_k = results["k_predict"]
            factors_path = os.path.join(f'{model.save_path_full}', f'WH_k={predict_k}.npz')
            factors_data = np.load(factors_path)
            current_node.W = factors_data["W"]
            current_node.H = factors_data["H"]
            current_node.k = predict_k
            current_node.factors_path = factors_path
        
        #
        # apply clustering
        #
        if self.cluster_on == "W":
            cluster_labels = np.argmax(current_node.W, axis=1)
            clusters = np.arange(0, current_node.W.shape[1], 1)

        elif self.cluster_on == "H":
            cluster_labels = np.argmax(current_node.H, axis=0)
            clusters = np.arange(0, current_node.H.shape[0], 1)

        # obtain the unique number of clusters that samples falls to
        n_clusters = len(set(cluster_labels))

        # leaf node based on depth limit or single cluster or all samples in same cluster
        if ((current_node.depth >= self.depth) and self.depth > 0) or current_node.k == 1 or n_clusters == 1:
            current_node.leaf = True 
            pickle_path = os.path.join(f'{node_save_path}', f'node_{current_node.node_name}.p')
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}
        
        #
        # go through each topic/cluster
        #
        for c in clusters:

            # current cluster samples
            cluster_c_indices = np.argwhere(cluster_labels == c).flatten()

            # empty cluster
            if len(cluster_c_indices) == 0:
                continue

            extracted_indicies = np.array([int(current_node.original_indices[i]) for i in cluster_c_indices])

            # save current results
            if self.random_identifiers:
                next_name = str(uuid.uuid1())
            else:
                next_name = f'{node_name}_{c}'
            current_node.child_node_names.append(next_name)
            
            # prepare next job
            next_job = {
                "parent_node_name":node_name,
                "node_name":next_name,
                "Ks":self._get_curr_Ks(node_k=current_node.k, num_samples=len(extracted_indicies)),
                "original_indices":extracted_indicies.copy(),
                "cluster_indices_in_parent":cluster_c_indices,
                "depth":current_node.depth+1,
                "parent_topic":int(c), 
                "parent_node_save_path":os.path.join(f'{node_save_path}', f'node_{current_node.node_name}.p'),
                "parent_node_factors_path":str(current_node.factors_path),
                "parent_node_k":int(current_node.k),
            }
            target_jobs.append(next_job)


        # save the node 
        pickle_path = os.path.join(f'{node_save_path}', f'node_{current_node.node_name}.p')
        pickle.dump(current_node, open(pickle_path, "wb"))

        return {"name":node_name, "target_jobs":target_jobs, "node_save_path":pickle_path}
    
    def _adjust_curr_Ks(self, X_shape, Ks):
        if max(Ks) >= min(X_shape):
            try:
                Ks = range(1, min(X_shape), self.Ks_deep_step)
            except Exception as e:
                print(e)
                return []
            
        return Ks
    
    def _get_curr_Ks(self, node_k, num_samples):
        if not self.K2:
            if self.Ks_deep_max is None:
                k_max = node_k + 1
            else:
                k_max = self.Ks_deep_max + 1

            new_max_K = min(k_max, min(num_samples, self.num_features))
            new_Ks = range(self.Ks_deep_min, new_max_K, self.Ks_deep_step)
        else:
            new_Ks = [2]
        
        return new_Ks

    def traverse_nodes(self):
        """
        Graph iterator. Returns all nodes in list format.\n
        This operation will load each node persistently into the memory.

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
    
    def traverse_tiny_leaf_topics(self, threshold=5):
        """
        Graph iterator with thresholding on number of documents. Returns a list of nodes where number of documents are less than the threshold.\n
        This operation is online, only the nodes that are outliers based on the number of documents are kept in the memory.
        
        Parameters
        ----------
        threshold : int
            Minimum number of documents each node should have.

        Returns
        -------
        data : list
            List of dictionarys that are format of node for each entry in the list.

        """
        self._all_nodes = []
        self._get_traversal(self.root, small_docs_thresh=threshold)
        return_data = self._all_nodes.copy()
        self._all_nodes = []

        return return_data
    
    def get_tiny_leaf_topics(self):
        """
        Graph iterator for tiny documents if processed already with self.process_tiny_leaf_topics(threshold:int).\n

        Returns
        -------
        tiny_leafs : list
            List of dictionarys that are format of node for each entry in the list.

        """
        try:
            return pickle.load(open(os.path.join(self.experiment_name, "tiny_leafs.p"), "rb"))
        except Exception as e:
            print("Could not load the tiny leafs. Did you call process_tiny_leaf_topics(threshold:int)?", e)
            return None
    
    def process_tiny_leaf_topics(self, threshold=5):
        """
        Graph post-processing with thresholding on number of documents.\n
        Returns a list of all tiny nodes, with all the nodes that had number of documents less than the threshold.\n
        Removes these outlier nodes from child-node lists on the original graph from their parents.\n
        Graph is re-set each time this function is called such that original child nodes are re-assigned.\n
        If threshold=None, this function will re-assign the original child indices only, and return None.

        Parameters
        ----------
        threshold : int
            Minimum number of documents each node should have.

        Returns
        -------
        tiny_leafs : list
            List of dictionarys that are format of node for each entry in the list.

        """
        
        # set the old child nodes on each node
        self._update_child_nodes_traversal(self.root)
       
        # remove the old saved tiny leafs 
        try:
            os.remove(os.path.join(self.experiment_name, "tiny_leafs.p"))
        except:
            pass

        # if threshold is none, we reversed everything
        if threshold is None:
            return
        
        tiny_leafs = self.traverse_tiny_leaf_topics(threshold=threshold)
        pickle.dump(tiny_leafs, open(os.path.join(self.experiment_name, "tiny_leafs.p"), "wb"))

        # remove tinly leafs from its parents
        for tf in tiny_leafs:
            my_name = tf["node_name"]
            parent_name = tf["parent_node_name"]
            parent_node = self._search_traversal(self.root, parent_name)

            # remove from online iterator
            parent_node.child_nodes = [node for node in parent_node.child_nodes if node.node_name != my_name]
            
            # also need to remove from saved node data
            parent_node_loaded = parent_node()
            parent_node_loaded.child_node_names = [node_name for node_name in parent_node_loaded.child_node_names if node_name != my_name]
            pickle.dump(parent_node_loaded, open(os.path.join(self.experiment_name, *parent_node_loaded.node_save_path.split(os.sep)[1:]), "wb"))

        return tiny_leafs
    
    def _update_child_nodes_traversal(self, node):
        
        for nn in node.original_child_nodes:
            self._update_child_nodes_traversal(nn)
        
        if node.child_nodes != node.original_child_nodes:
            node.child_nodes = node.original_child_nodes

        node_loaded = node()
        if node_loaded.original_child_node_names != node_loaded.child_node_names:
            node_loaded.child_node_names = node_loaded.original_child_node_names
            pickle.dump(node_loaded, open(os.path.join(self.experiment_name, *node_loaded.node_save_path.split(os.sep)[1:]), "wb"))
    
    def _search_traversal(self, node, name):
        
        # Base case: if the current node matches the target name
        if node.node_name == name:
            return node

        # Recursive case: search the children
        for nn in node.child_nodes:
            result = self._search_traversal(nn, name)
            
            # If the recursive search found the node, return it
            if result is not None:
                return result
        
        # If the node is not found in this branch, return None
        return None

    def _get_traversal(self, node, small_docs_thresh=None):

        for nn in node.child_nodes:
            self._get_traversal(nn, small_docs_thresh=small_docs_thresh)

        if small_docs_thresh is not None:
            tmp_node_data = vars(node()).copy()
            if not (tmp_node_data["leaf"] and tmp_node_data["num_samples"] < small_docs_thresh):
                return

        data = vars(node()).copy()
        data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
        if data["node_name"] != self.root_name:
            data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
            data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])

        if data["factors_path"] is not None:
            data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])

        self._all_nodes.append(data)

    def go_to_root(self):
        """
        Graph iterator. Goes to root node.\n
        This operation is online, only one node is kept in the memory at a time.

        Returns
        -------
        data : dict
            Dictionary format of node.

        """
        self.iterator = self.root
        data = vars(self.iterator()).copy()
        data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
        if data["node_name"] != self.root_name:
            data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
            data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])

        if data["factors_path"] is not None:
            data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])

        return data
    
    def go_to_node(self, name:str):
        """
        Graph iterator. Goes to node specified by name (node.node_name).\n
        This operation is online, only one node is kept in the memory at a time.

        Parameters
        ----------
        name : str
            Name of the node

        Returns
        -------
        data : dict
            Dictionary format of node.

        """
        result = self._search_traversal(node=self.root, name=name)
        if result is not None:
            self.iterator = result
            data = vars(self.iterator()).copy()
            data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
            
            if data["node_name"] != self.root_name:
                data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
                data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])

            if data["factors_path"] is not None:
                data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])
            
            return data
        else:
            raise Exception("Node not found!")
        

    def get_node(self):
        """
        Graph iterator. Returns the current node.\n
        This operation is online, only one node is kept in the memory at a time.

        Returns
        -------
        data : dict
            Dictionary format of node.

        """
        data = vars(self.iterator()).copy()
        data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
        if data["node_name"] != self.root_name:
            data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
            data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])

        if data["factors_path"] is not None:
            data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])

        return data

    def go_to_parent(self):
        """
        Graph iterator. Goes to the parent of current node.\n
        This operation is online, only one node is kept in the memory at a time.

        Returns
        -------
        data : dict
            Dictionary format of node.
        """
        if self.iterator is not None:
            self.iterator = self.iterator.parent_node
            data = vars(self.iterator()).copy()
            data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
            if data["node_name"] != self.root_name:
                data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
                data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])

            if data["factors_path"] is not None:
                data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])

            return data
        else:
            print("At the root! There is no parents.")

    def go_to_children(self, idx: int):
        """
        Graph iterator. Goes to the child node specified by index.\n
        This operation is online, only one node is kept in the memory at a time.

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
            data = vars(self.iterator()).copy()
            data["node_save_path"] = os.path.join(self.experiment_name, *data["node_save_path"].split(os.sep)[1:])
            if data["node_name"] != self.root_name:
                data["parent_node_save_path"] = os.path.join(self.experiment_name, *data["parent_node_save_path"].split(os.sep)[1:])
                data["parent_node_factors_path"] = os.path.join(self.experiment_name, *data["parent_node_factors_path"].split(os.sep)[1:])
            
            if data["factors_path"] is not None:
                data["factors_path"] = os.path.join(self.experiment_name, *data["factors_path"].split(os.sep)[1:])

            return data

        except:
            print("Children at index "+str(idx)+" from the current does not exist!")

    def _organize_nmfk_params(self, params):

        #
        # Required
        #
        params["collect_output"] = False
        params["save_output"] = True
        params["n_nodes"] = 1
        
        if not self.K2:
            params["predict_k"] = True


        return params
    
    def _process_results(self, node_results, save_checkpoint):
        # remove the job
        del self.target_jobs[node_results["name"]]

        # save node save paths
        self.node_save_paths[node_results["name"]] = node_results["node_save_path"]

        # save
        for next_job in node_results["target_jobs"]:
            self.target_jobs[next_job["node_name"]] = next_job

        # checkpointing
        if save_checkpoint:
            self._save_checkpoint()

    def _load_checkpoint_file(self):
        try:
            saved_class_params = pickle.load(
                open(os.path.join(self.experiment_save_path, "checkpoint.p"), "rb")
            )
            return saved_class_params
        except Exception:
            warnings.warn("No checkpoint file found!")
            return {}
            
    def _load_checkpoint(self, saved_class_params):
        if self.verbose:
            print("Loading saved object state from checkpoint...")

        self._set_params(saved_class_params)

    def _set_params(self, class_parameters):
        """Sets class variables from the loaded checkpoint"""
        for parameter, value in class_parameters.items():
            setattr(self, parameter, value)

    def _save_checkpoint(self):
        class_params = vars(self).copy()
        del class_params["X"]
        if self.generate_X_callback is not None:
            del class_params["generate_X_callback"]

        pickle.dump(class_params, open(os.path.join(
            self.experiment_save_path, "checkpoint.p"), "wb"))
