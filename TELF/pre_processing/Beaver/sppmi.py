import numpy as np
import scipy


def sppmi(cooc, shift=4):
    """computes the shifted positive pointwise mutual information from the cooccurrence matrix
    input:
      cooc: sparse cooccurence matrix
      shift: the shift
    output:
      sppmi_matrix: sparse shifted positive cooccurrence matrix

    author: Erik Skau
    """
    total = np.sum(cooc)
    colsum = np.array(np.sum(cooc, axis=0))[0, :].astype(np.float64)
    rowsum = np.array(np.sum(cooc, axis=1))[:, 0].astype(np.float64)
    colsuminv = np.divide(
        np.ones_like(colsum), colsum, out=np.zeros_like(colsum), where=colsum != 0
    )
    rowsuminv = np.divide(
        np.ones_like(rowsum), rowsum, out=np.zeros_like(rowsum), where=rowsum != 0
    )
    sppmi_matrix = scipy.sparse.diags(rowsuminv) * cooc * scipy.sparse.diags(colsuminv)
    sppmi_matrix.data = np.maximum(np.log(sppmi_matrix.data * total) - shift, 0)
    sppmi_matrix.eliminate_zeros()
    return sppmi_matrix
