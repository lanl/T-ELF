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
import sys
import pickle
import warnings

try:
    from mpi4py import MPI
except:
    MPI = None


class OnlineNode():
    def __init__(self, 
                 node_path:str,
                 parent_node:object,
                 child_nodes:list,
                 ) -> None:
        self.node_path = node_path
        self.parent_node = parent_node
        self.child_nodes = child_nodes
        self.node_data = None
    
    def __call__(self, persistent=False):

        if persistent:
            if self.node_data is None:
                 self.node_data = pickle.load(open(self.node_path, "rb"))
            
            return self.node_data

        else:
            return pickle.load(open(self.node_path, "rb"))


class Node():
    def __init__(self,
                 node_name: str,
                 depth: int,
                 parent_topic: int,
                 W: np.ndarray, H: np.ndarray, k: int,
                 parent_node_name: str,
                 child_node_names: list,
                 original_indices: np.ndarray,
                 num_samples:int,
                 leaf:bool,
                 user_node_data:dict
                 ):
        self.node_name = node_name
        self.depth = depth
        self.W = W
        self.H = H
        self.k = k
        self.parent_topic = parent_topic
        self.parent_node_name = parent_node_name
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
                 generate_X_callback=None,
                 n_nodes=1,
                 verbose=True,
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


    def fit(self, X, Ks, from_checkpoint=False, save_checkpoint=False):
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
            If True, it saves checkpoints. The default is False.
        
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
            self.node_status = {}
            for ii in range(1, self.n_nodes, 1):
                self.node_status[ii] = {"free":True, "job":None}

        # single node processing
        else:
            comm = None
            rank = 0
            self.node_status[0] = {"free": True, "job":None}


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
                    self.num_features = X.shape[1]

                elif self.cluster_on == "H":
                    original_indices = np.arange(0, X.shape[1], 1)
                    self.num_features = X.shape[0]

                self.root_name = str(uuid.uuid1())
                self.target_jobs[self.root_name] = {
                    "parent_node_name":"None",
                    "node_name":self.root_name,
                    "Ks":Ks,
                    "original_indices":original_indices,
                    "depth":0,
                    "parent_topic":None
                }

        # save data matrix
        self.X = X 

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
                    self._signal_workers_exit(comm)
                break
            
            # worker nodes check exit status
            if self.n_nodes > 1:
                self._worker_check_exit_status(rank, comm)

            # recieve jobs from rank 0 at worker ndoes
            elif rank != 0 and self.n_nodes > 1:
                job_data, job_flag = self._get_next_job_at_worker(
                    rank, comm)

            # single node job schedule
            else:
                available_jobs = list(self.target_jobs.keys())
                next_job = available_jobs.pop(0)
                job_data = self.target_jobs[next_job]

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
                all_node_results = self._collect_results_from_workers(rank, comm)
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
            parent_node=None,
            child_nodes=[]
        )
        self.iterator = self.root
        self._prepare_iterator(self.root)

        return {"time":total_exec_seconds}


    def _prepare_iterator(self, node):
        
        node_object = pickle.load(open(node.node_path, "rb"))
        child_node_names = node_object.child_node_names

        for child_name in child_node_names:
            child_node = OnlineNode(
                node_path=self.node_save_paths[child_name],
                parent_node=node,
                child_nodes=[])
            node.child_nodes.append(child_node)
            self._prepare_iterator(child_node)

        return
        
    def _process_node(self, Ks, depth, original_indices, node_name, parent_node_name, parent_topic):

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
        current_node = Node(
                node_name=node_name,
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
        )

        #
        # check if leaf node status
        #
        if (current_node.num_samples == 1) or (self.sample_thresh > 0 and (current_node.num_samples <= self.sample_thresh)):
            current_node.leaf = True
            pickle_path = f'{node_save_path}/node_{current_node.node_name}.p'
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
        # prepare the current nmfk parameters
        #
        if current_node.depth >= len(self.nmfk_params):
            select_params = -1
        else:
            select_params = current_node.depth 
        curr_nmfk_params = self.nmfk_params[select_params % len(self.nmfk_params)]
        curr_nmfk_params["save_path"] = node_save_path

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
            pickle_path = f'{node_save_path}/node_{current_node.node_name}.p'
            pickle.dump(current_node, open(pickle_path, "wb"))
            return {"name":node_name, "target_jobs":[], "node_save_path":pickle_path}

        #
        # latent factors
        #
        if self.K2:
            factors_data = np.load(f'{model.save_path_full}/WH_k=2.npz')
            current_node.W = factors_data["W"]
            current_node.H = factors_data["H"]
            current_node.k = 2
        
        else:
            predict_k = results["k_predict"]
            factors_data = np.load(f'{model.save_path_full}/WH_k={predict_k}.npz')
            current_node.W = factors_data["W"]
            current_node.H = factors_data["H"]
            current_node.k = predict_k
        
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

        # leaf node or single cluster or all samples in same cluster
        if ((current_node.depth >= self.depth) and self.depth > 0) or current_node.k == 1 or n_clusters == 1:
            current_node.leaf = True
            pickle_path = f'{node_save_path}/node_{current_node.node_name}.p'
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

            # save current results
            next_name = str(uuid.uuid1())
            current_node.child_node_names.append(next_name)
            
            # prepare next job
            next_job = {
                "parent_node_name":node_name,
                "node_name":next_name,
                "Ks":self._get_curr_Ks(node_k=current_node.k, num_samples=len(cluster_c_indices)),
                "original_indices":cluster_c_indices.copy(),
                "depth":current_node.depth+1,
                "parent_topic":c,
            }
            target_jobs.append(next_job)


        # save the node
        pickle_path = f'{node_save_path}/node_{current_node.node_name}.p'
        pickle.dump(current_node, open(pickle_path, "wb"))

        return {"name":node_name, "target_jobs":target_jobs, "node_save_path":pickle_path}
    
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

    def _get_traversal(self, node):

        for nn in node.child_nodes:
            self._get_traversal(nn)

        data = vars(node(persistent=True)).copy()
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
        return data

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

    def _signal_workers_exit(self, comm):
        for job_rank in range(1, self.n_nodes, 1):
            req = comm.isend(np.array([True]),
                             dest=job_rank, tag=int(f'400{job_rank}'))
            req.wait()

    def _worker_check_exit_status(self, rank, comm):
        if comm.iprobe(source=0, tag=int(f'400{rank}')):
            sys.exit(0)

    def _get_next_job_at_worker(self, rank, comm):
        job_flag = True
        if comm.iprobe(source=0, tag=int(f'200{rank}')):
            req = comm.irecv(buf=bytearray(b" " * self.comm_buff_size),
                             source=0, tag=int(f'200{rank}'))
            data = req.wait()

        else:
            job_flag = False
            data = {}

        return data, job_flag

    def _collect_results_from_workers(self, rank, comm):

        all_results = []
        # collect results at root
        if self.n_nodes > 1 and rank == 0:
            for job_rank, status_info in self.node_status.items():
                if self.node_status[job_rank]["free"] == False and comm.iprobe(source=job_rank, tag=int(f'300{job_rank}')):
                    req = comm.irecv(buf=bytearray(b" " * self.comm_buff_size),
                                     source=job_rank, tag=int(f'300{job_rank}'))
                    node_results = req.wait()
                    self.node_status[job_rank]["free"] = True
                    self.node_status[job_rank]["job"] = None
                    all_results.append(node_results)

        return all_results
    
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
        pickle.dump(class_params, open(os.path.join(
            self.experiment_save_path, "checkpoint.p"), "wb"))
