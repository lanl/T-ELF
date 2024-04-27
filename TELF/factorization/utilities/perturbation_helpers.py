
from ..decompositions.utilities.resample import poisson, uniform_product

#
# Matrix operations
#
def perturb_X(X, perturbation:int, epsilon:float, perturb_type:str):

    if perturb_type == "uniform":
        Y = uniform_product(X, epsilon, random_state=perturbation)
    elif perturb_type == "poisson":
        Y = poisson(X, random_state=perturbation)

    return Y

#
# Tensor operations
#
def perturb_tensor_X(X, perturbation:int, epsilon:float, perturb_type:str):

    if perturb_type == "uniform":
        Y = [uniform_product(X_, epsilon, random_state=perturbation) for X_ in X]
    elif perturb_type == "poisson":
        Y = [poisson(X_, random_state=perturbation) for X_ in X]

    return Y
