from TELF.factorization import NMFk
import mat73
import pytest

def test_wtsi_k():
    
    X = mat73.loadmat('../data/wtsi.mat')['X'].astype('float32')

    params = {
        "n_perturbs":32,
        "n_iters":5000,
        "epsilon":0.015,
        "n_jobs":-1,
        "init":"nnsvd",
        "use_gpu":False,
        "save_output":False,
        "collect_output":True,
        "predict_k":True,
        "verbose":True,
        "transpose":False,
        "sill_thresh":0.9,
        "nmf_verbose":False,
        "nmf_method":'nmf_kl_mu',
        "k_search_method":"bst_post",
        "H_sill_thresh":0.1
    }
    model = NMFk(**params)
    
    Ks = range(1,8,1)
    name = "wtsi"
    note = "This is an example run of NMFk"
    results = model.fit(X, Ks, name, note)
    assert results['k_predict'] == 4
