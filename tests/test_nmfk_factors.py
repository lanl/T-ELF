from TELF.factorization import NMFk
import pytest
import numpy as np
import scipy.sparse as ss
from scipy import optimize
from TELF.factorization.decompositions.utilities.math_utils import fro_norm

def computErr_np(Worig,Wfin): #Computes the error in factor matrices for given np array.
    if Worig.shape!=Wfin.shape:
        raise('W matricies not of the shame shape')
    #Normalize features of W matricies
    WorigN = Worig * ((1/np.linalg.norm(Worig,axis=0))[np.newaxis,:])
    WfinN = Wfin * ((1/np.linalg.norm(Wfin,axis=0))[np.newaxis,:])
    #compute a matrix of errors 
    errMat = np.linalg.norm(WorigN[:,np.newaxis,:]-WfinN[:,:,np.newaxis],axis=0)
    #Squaring this matrix allows us to add feature errors together to minimize the sum of the total matrix error. 
    errMat = errMat**2
    #Locate combination that results in lowest error
    order = optimize.linear_sum_assignment(errMat)[1]
    #Reorder Matrix. I do not know if it will be possible to maintain gradient information through a reorder operation in Pytorch. This is your task
    WfinNS = WfinN[:,order]
    #Compute Error
    error = np.linalg.norm(WorigN-WfinNS)/np.linalg.norm(WorigN)
    return error

def test_nmfk_factors():
    
    # load data
    W_true = np.load("../data/test/Wtrue.npz", allow_pickle=True)["W"]
    W_compare = np.load("../data/test/Wcompare.npz")["W"]
    H_compare = np.load("../data/test/Hcompare.npz")["H"]
    X = ss.load_npz("../data/test/Xtrue.npz")
    
    print(X.shape, W_true.shape, W_compare.shape, H_compare.shape)
    
    # run nmfk
    params = {
        "n_perturbs":32,
        "n_iters":1500,
        "epsilon":0.001,
        "n_jobs":-1,
        "init":"nnsvd", 
        "use_gpu":False,
        "save_output":False,
        "collect_output":True,
        "predict_k_method":"sill",
        "verbose":True,
        "nmf_verbose":False,
        "transpose":False,
        "sill_thresh":0.8,
        "pruned":True,
        'nmf_method':'nmf_fro_mu',
        "calculate_error":True,
        "predict_k":True,
        "use_consensus_stopping":0,
        "calculate_pac":True,
        "consensus_mat":True,
        "perturb_type":"uniform",
        "perturb_multiprocessing":False,
        "perturb_verbose":False,
        "simple_plot":True
    }
    model = NMFk(**params)
    Ks = range(3,7,1)
    name = "test_factors"
    results = model.fit(X, Ks, name, "")
    W, H = results["W"], results["H"]
    
    # check for correctness
    assert results['k_predict'] == 5
    assert computErr_np(W, W_compare) < 1e-4
    
    
    
