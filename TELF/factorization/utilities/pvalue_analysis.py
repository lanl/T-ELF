import numpy as np
from scipy.stats import wilcoxon


def pvalue_analysis(errRegres, Ks, SILL_MIN, SILL_thr=0.9):
    """


    Parameters
    ----------
    errRegres : TYPE
        DESCRIPTION.
    Ks : TYPE
        DESCRIPTION.
    SILL_MIN : TYPE
        DESCRIPTION.
    SILL_thr : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if len(Ks) == 1:
        return Ks[0], np.ones(len(Ks))

    pvalue = np.ones(len(Ks))
    oneDistrErr = errRegres[0]
    i = 1
    nopt = 0
    
    # if flat sill score
    if SILL_MIN[-1] >= SILL_thr:
        return Ks[-1], pvalue
    
    # if not flat do the regular way
    while i < (len(Ks)):

        if SILL_MIN[i] > SILL_thr:  # 0.75:
            pvalue[i-1] = wilcoxon(oneDistrErr, errRegres[i])[1]
            if pvalue[i-1] < 0.05:
                nopt = i
                oneDistrErr = np.copy(errRegres[i])
                i = i + 1
            else:
                i = i + 1
        else:
            i = i + 1

    Ks = list(Ks)
    
    return Ks[nopt], pvalue

