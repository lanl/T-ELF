from .NMFk import NMFk
from .decompositions.nmf_fro_mu import H_update as H_update_fro_mu
from .decompositions.nmf_kl_mu import H_update as H_update_kl_mu
from ..helpers.inits import organize_required_params
from .utilities.math import MitH
from sklearn.svm import SVR
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sp_hstack
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

try:
    import cupy as cp
except Exception:
    cp = None


class SPLITTransfer():

    def __init__(self,
                 Ks_known,
                 Ks_target,
                 Ks_split_step=1,
                 Ks_split_min=1,
                 H_regress_gpu=False,
                 H_learn_method="regress",
                 nmfk_params_known={},
                 nmfk_params_target={},
                 nmfk_params_split={},
                 H_regress_iters=1000,
                 H_regress_method="fro",
                 H_regress_init="random",
                 transfer_regress_params={},
                 transfer_method="SVR",
                 transfer_model=None,
                 verbose=True,
                 random_state=42):

        np.random.seed(random_state)
        self.Ks_known = Ks_known
        self.Ks_target = Ks_target
        self.Ks_split = None
        self.Ks_split_step = Ks_split_step
        self.Ks_split_min = Ks_split_min
        self.H_regress_gpu = H_regress_gpu
        self.H_learn_method = H_learn_method
        self.required_params_setting = {
            "collect_output":True,
            "predict_k":True,
        }
        self.nmfk_params_known = organize_required_params(nmfk_params_known, self.required_params_setting)
        self.nmfk_params_target = organize_required_params(nmfk_params_target, self.required_params_setting)
        self.nmfk_params_split = organize_required_params(nmfk_params_split, self.required_params_setting)
        self.H_regress_iters = H_regress_iters
        self.H_regress_method = H_regress_method
        self.H_regress_init = H_regress_init
        self.transfer_method = transfer_method
        self.transfer_regress_params = transfer_regress_params
        self.verbose = verbose
        self.transfer_model = transfer_model

        #
        # Parameter check
        #
        H_learn_avail = ["regress", "MitH"]
        assert self.H_learn_method in H_learn_avail, "Unknown H learn method! Choose from: " + \
            ", ".join(H_learn_avail)

        H_init_avail = ["random", "MitH"]
        assert self.H_regress_init in H_init_avail, "Unknown H learn method! Choose from: " + \
            ", ".join(H_learn_avail)

        #
        # Setup NMFk and set variables
        #
        self.known = {}
        self.known["nmfk"] = NMFk(**self.nmfk_params_known)
        self.known["X"] = None
        self.known["W"] = None
        self.known["H"] = None
        self.known["H_learned"] = None
        self.known["k"] = None

        self.target = {}
        self.target["nmfk"] = NMFk(**self.nmfk_params_target)
        self.target["X"] = None
        self.target["W"] = None
        self.target["H"] = None
        self.target["H_learned"] = None
        self.target["k"] = None

        self.split = {}
        self.split["nmfk"] = NMFk(**self.nmfk_params_split)
        self.split["X"] = None
        self.split["W"] = None
        self.split["H"] = None
        self.split["M_known"] = None
        self.split["M_target"] = None
        self.split["k"] = None

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

        #
        # Setup learner
        #
        if self.transfer_method == "SVR":
            self.transfer_model = SVR(**transfer_regress_params)

        elif self.transfer_method == "model":
            assert transfer_model != None, "Pass transfer model with transfer_model parameter!"
            self.transfer_model = transfer_model(**transfer_regress_params)

        else:
            raise Exception("Unknown transfer method!")

        #
        # Other variables
        #
        self.train_inds = None
        self.test_inds = None

    # ====================================================
    # Public Functions
    # ====================================================

    def fit_transform(self, X_known, X_target, indicator: np.ndarray):
        _ = self.fit(X_known, X_target)
        _ = self.transform(indicator)

        return self

    def fit(self, X_known, X_target):

        self.fit_known(X_known)
        self.fit_target(X_target)
        self.fit_split()

        return self

    def transform(self, indicator: np.ndarray):

        # learn H
        self.learn_H()

        # transfer learning
        self.fit_transfer(indicator)

        return self

    def predict(self, test=True):

        if test:
            predict = self.transfer_model.predict(self.target["H_learned"].T)
            predict[~self.test_inds] = np.nan

        else:
            predict = self.transfer_model.predict(self.known["H_learned"].T)

            if not all(~self.train_inds == False):
                predict[~self.train_inds] = np.nan

        return predict

    def get_feature_importances(self, indicator, permi_params={}, feature_names=[], plot=True, rotate_xticks=False):

        if "n_repeats" not in permi_params:
            permi_params["n_repeats"] = 30

        if "random_state" not in permi_params:
            permi_params["random_state"] = 42

        r = permutation_importance(
            self.transfer_model,
            self.known["H_learned"].T[self.train_inds, :],
            indicator[self.train_inds],
            **permi_params
        )

        feature_importances = {}
        feature_indices_sorted = r.importances_mean.argsort()[::-1]
        indices = []
        for ii in feature_indices_sorted:
            if r.importances_mean[ii] - 2 * r.importances_std[ii] > 0:
                indices.append(ii)
                if len(feature_names) == len(feature_indices_sorted):
                    fname = feature_names[ii]
                else:
                    fname = "Feature " + str(ii + 1)

                if self.verbose:
                    print(fname)
                    print("Importance Mean=", f"{r.importances_mean[ii]:.4f}")
                    print("Importance STD=", f" +/- {r.importances_std[ii]:.4f}")
                    print("--------------------------------")

                feature_importances[fname] = {
                    "importance_mean": round(r.importances_mean[ii], 4),
                    "importance_std": round(r.importances_std[ii], 4),
                }

        if plot:
            self._plot_feature_importances(feature_importances, rotate_xticks)

        feature_importances["indices"] = indices
        return feature_importances

    def get_score(self, indicator):
        score = self.transfer_model.score(
            self.known["H_learned"].T[self.train_inds],
            indicator[self.train_inds]
        )

        if self.verbose:
            print("R-squared=", score)

        return score

    def fit_known(self, X_known):
        self.known["nmfk"] = NMFk(**self.nmfk_params_known)

        # fit known
        if self.verbose:
            print("Fitting known set...")
        self.known["X"] = X_known
        nmfk_results_known = self.known["nmfk"].fit(self.known["X"], self.Ks_known, name="KNOWN")
        self.known["k"] = nmfk_results_known["k_predict"]
        self.known["W"] = nmfk_results_known["W"]
        self.known["H"] = nmfk_results_known["H"]

        return self

    def fit_target(self, X_target):
        self.target["nmfk"] = NMFk(**self.nmfk_params_target)

        # fit target
        if self.verbose:
            print("Fitting target set...")
        self.target["X"] = X_target
        nmfk_results_target = self.target["nmfk"].fit(self.target["X"], self.Ks_target, name="TARGET")
        self.target["k"] = nmfk_results_target["k_predict"]
        self.target["W"] = nmfk_results_target["W"]
        self.target["H"] = nmfk_results_target["H"]

        return self

    def fit_split(self):
        self.split["nmfk"] = NMFk(**self.nmfk_params_split)

        if self.verbose:
            print("Fitting split set...")
        self.split["X"] = self._get_split_X([self.known["W"], self.target["W"]])
        self.Ks_split = self._get_split_Ks()
        nmfk_results_split = self.split["nmfk"].fit(self.split["X"], self.Ks_split, name="SPLIT")
        self.split["k"] = nmfk_results_split["k_predict"]
        self.split["W"] = nmfk_results_split["W"]
        self.split["H"] = nmfk_results_split["H"]
        self.split["M_known"] = self.split["H"][:, 0:self.known["k"]].copy()
        self.split["M_target"] = self.split["H"][:, self.known["k"]:].copy()

        return self

    def fit_transfer(self, indicator):
        if self.verbose:
            print("Applying transfer learning...")
        self.transfer_model.fit(self.known["H_learned"].T[self.train_inds, :], indicator[self.train_inds])

    def learn_H(self):

        # Apply regression
        if self.H_learn_method == "regress":
            k = self.split["k"]

            if self.verbose:
                print("Applying regression to get known H learned...")

            H = self._init_H_regress(self.known["X"], self.split["M_known"], self.known["H"], k)

            self.known["H_learned"] = self._H_regression(self.known["X"], self.split["W"], H)

            if self.verbose:
                print("Applying regression to get target H learned...")

            H = self._init_H_regress(self.target["X"], self.split["M_target"], self.target["H"], k)

            self.target["H_learned"] = self._H_regression(self.target["X"], self.split["W"], H)

        # multiply Mi with H known or target
        elif self.H_learn_method == "MitH":
            if self.verbose:
                print("Learning known H with MitH...")
            self.known["H_learned"] = MitH(self.split["M_known"], self.known["H"])

            if self.verbose:
                print("Learning target H with MitH...")
            self.target["H_learned"] = MitH(self.split["M_target"], self.target["H"])

        else:
            raise Exception("Unknown H learn method!")

        # get train test indices
        self.train_inds = np.any(self.known["H_learned"].T != 0, axis=1)
        self.test_inds = np.any(self.target["H_learned"].T != 0, axis=1)

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
            return np.random.rand(k, X.shape[1])

        elif self.H_regress_init == "MitH":
            return MitH(M, H)

        else:
            raise Exception("Unknown H regression initilization!")

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

    def _get_split_Ks(self):
        max_k = self.known["k"] + self.target["k"]
        return range(self.Ks_split_min, max_k, self.Ks_split_step)

    def _get_split_X(self, Ws: list):
        if issparse(Ws[0]):
            return sp_hstack(Ws)
        else:
            return np.hstack(Ws)

    def _plot_feature_importances(self, feature_importances, rotate_xticks=False):

        feature_importances = feature_importances.copy()

        if len(feature_importances) > 0:
            plt.figure(dpi=100)

            names = []
            importance_means = []
            importance_stds = []

            for key, values in feature_importances.items():
                names.append(key)
                importance_means.append(values["importance_mean"])
                importance_stds.append(values["importance_std"])

            plt.bar(names, importance_means, yerr=importance_stds,
                    align='center',
                    alpha=0.5,
                    ecolor='black',
                    capsize=10,
                    )

            plt.title("Feature Importances")
            plt.ylabel("Importance Mean")
            plt.xlabel("Features")

            if rotate_xticks:
                plt.xticks(rotation=90)

            plt.show()
        else:
            print("All feature importances were more than 2 std away from mean!")
