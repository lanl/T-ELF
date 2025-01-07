
from ..decompositions.utilities.resample import poisson, uniform_product, boolean

#
# Matrix operations
#
def perturb_X(X, perturbation:int, epsilon:float, perturb_type:str):

    if perturb_type == "uniform":
        Y = uniform_product(X, epsilon=epsilon, random_state=perturbation)
    elif perturb_type == "poisson":
        Y = poisson(X, random_state=perturbation)
    elif perturb_type == "bool" or perturb_type == "boolean":
        Y = boolean(X, epsilon=epsilon, random_state=perturbation)
    else:
        raise Exception("Unknown perturbation type")

    return Y

#
# Tensor operations
#
def perturb_tensor_X(X, perturbation:int, epsilon:float, perturb_type:str):

    if perturb_type == "uniform":
        Y = [uniform_product(X_, epsilon, random_state=perturbation) for X_ in X]
    elif perturb_type == "poisson":
        Y = [poisson(X_, random_state=perturbation) for X_ in X]
    else:
        raise Exception("Unknown perturbation type")

    return Y
