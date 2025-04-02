from .NMFk import NMFk
from .decompositions.nmf_fro_mu import H_update as H_update_fro_mu
from .decompositions.nmf_kl_mu import H_update as H_update_kl_mu

import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sp_hstack

try:
    import cupy as cp
except Exception:
    cp = None


class SPLIT():

    def __init__(self,
                 Xs: dict,
                 Ks: dict,
                 nmfk_params: dict,
                 split_nmfk_params={},
                 Ks_split_step=1,
                 Ks_split_min=1,
                 H_regress_gpu=False,
                 H_learn_method="regress",
                 H_regress_iters=1000,
                 H_regress_method="fro",
                 H_regress_init="random",
                 verbose=True,
                 random_state=42
                 ):

        self.Ks_split = None
        self.Ks_split_step = Ks_split_step
        self.Ks_split_min = Ks_split_min
        self.H_regress_gpu = H_regress_gpu
        self.H_learn_method = H_learn_method
        self.H_regress_iters = H_regress_iters
        self.H_regress_method = H_regress_method
        self.H_regress_init = H_regress_init
        self.verbose = verbose
        self.random_state = random_state
        self.split_nmfk_params = self._organize_nmfk_params(split_nmfk_params)
        self.return_data = None

        #
        # Prepare data and NMFk
        #

        # split information
        self.split_information = {
            "Ks": None,
            "nmfk_params": self.split_nmfk_params,
            "nmfk": NMFk(**self.split_nmfk_params),
            "X": None,
            "W": None,
            "H": None,
            "k": None
        }

        # user passed information
        self.information = {}

        for num, (name, X) in enumerate(Xs.items()):
            self.information[name] = {}

            # data
            self.information[name]["data"] = X

            # NMFk parameters
            if name in nmfk_params:
                self.information[name]["nmfk_params"] = self._organize_nmfk_params(nmfk_params[name])
            else:
                self.information[name]["nmfk_params"] = self._organize_nmfk_params({})

            # NMFk object
            self.information[name]["nmfk"] = NMFk(**self.information[name]["nmfk_params"])

            # Ks
            self.information[name]["Ks"] = Ks[name]

            # Other information
            self.information[name]["W"] = None
            self.information[name]["H"] = None
            self.information[name]["k"] = None
            self.information[name]["H_learned"] = None
            self.information[name]["M"] = None
            self.information[name]["num"] = num

        #
        # Parameter check
        #
        H_learn_avail = ["regress", "MitH"]
        assert self.H_learn_method in H_learn_avail, "Unknown H learn method! Choose from: " + \
            ", ".join(H_learn_avail)

        #
        # Setup H Regression Method
        #
        self.H_regress_opts = {"niter": self.H_regress_iters}

        if self.H_regress_method == "fro":
            self.H_regress_func = H_update_fro_mu

        elif self.H_regress_method == "kl":
            self.H_regress_func = H_update_kl_mu

        else:
            raise Exception("Unknown H regression method selected.")

    # ====================================================
    # Public Functions
    # ====================================================

    def fit(self, fname=""):

        #
        # APPLY NMFk ON Xs
        #
        information = self.information.copy()
        for name, info in information.items():

            if self.verbose:
                print("Applying NMFk:", name)

            X = info["data"]
            Ks = info["Ks"]
            nmfk_esults = info["nmfk"].fit(X, Ks, name=str(name)+"_"+str(fname))

            self.information[name]["k"] = nmfk_esults["k_predict"]
            self.information[name]["W"] = nmfk_esults["W"]
            self.information[name]["H"] = nmfk_esults["H"]

        #
        # APPLY SPLIT
        #
        if self.verbose:
            print("Applying SPLIT NMFk")

        self.split_information["X"] = self._get_split_X()
        self.split_information["Ks"] = self._get_split_Ks()
        split_nmfk_results = self.split_information["nmfk"].fit(self.split_information["X"],
                                                                self.split_information["Ks"], name="SPLIT_"+str(fname))

        self.split_information["k"] = split_nmfk_results["k_predict"]
        self.split_information["W"] = split_nmfk_results["W"]
        self.split_information["H"] = split_nmfk_results["H"]

        return self

    def transform(self):

        # LEARN H
        information = self.information.copy()
        for name, info in information.items():

            if self.verbose:
                print("Learning H:", name)

            self.information[name]["M"] = self._get_M(info["num"], info["k"])

            # Apply regression
            if self.H_learn_method == "regress":

                H = self._init_H_regress(info["data"], self.information[name]
                                         ["M"], info["H"], self.split_information["k"])
                self.information[name]["H_learned"] = self._H_regression(
                    info["data"], self.split_information["W"], H)

            # multiply Mi with H known or target
            elif self.H_learn_method == "MitH":

                self.information[name]["H_learned"] = self._MitH(self.information[name]["M"], info["H"])

            else:
                raise Exception("Unknown H learn method!")

        # ORGANIZE RETURN DATA
        return_data = {}
        for name, info in self.information.items():
            return_data[name] = {}
            return_data[name]["W"] = self.split_information["W"]
            return_data[name]["H"] = info["H_learned"]

        self.return_data = return_data

        return return_data

    def fit_transform(self):
        self.fit()
        return self.transform()

    def set_params(self, parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return vars(self)

    # ====================================================
    # Private Functions
    # ====================================================

    def _init_H_regress(self, X, M, H, k):

        if self.H_regress_init == "random":
            np.random.seed(self.random_state)
            return np.random.rand(k, X.shape[1])

        elif self.H_regress_init == "MitH":
            return self._MitH(M, H)

        else:
            raise Exception("Unknown H regression initilization!")

    def _MitH(self, M, H):
        return M @ H

    def _H_regression(self, X, W, H):

        H_learned_ = self.H_regress_func(
            X=cp.array(X) if self.H_regress_gpu else X,
            W=cp.array(W) if self.H_regress_gpu else W,
            H=cp.array(H) if self.H_regress_gpu else H,
            opts=self.H_regress_opts,
            use_gpu=self.H_regress_gpu)

        if self.H_regress_gpu:
            H_learned = cp.asnumpy(H_learned_)
            del H_learned_
            cp._default_memory_pool.free_all_blocks()
            return H_learned

        else:
            return H_learned_

    def _get_M(self, num, k):

        all_K = []
        for name, info in self.information.items():
            all_K.append(info["k"])

        return self.split_information["H"][:, sum(all_K[:num]):k+sum(all_K[:num])].copy()

    def _get_split_Ks(self):
        all_K = []
        for name, info in self.information.items():
            all_K.append(info["k"])

        max_k = sum(all_K) - 1
        return range(self.Ks_split_min, max_k, self.Ks_split_step)

    def _get_split_X(self):

        Ws = []
        for name, info in self.information.items():
            Ws.append(info["W"])

        if issparse(Ws[0]):
            return sp_hstack(Ws)
        else:
            return np.hstack(Ws)

    def _organize_nmfk_params(self, params):
        #
        # Required
        #
        params["collect_output"] = True
        params["predict_k"] = True

        return params
