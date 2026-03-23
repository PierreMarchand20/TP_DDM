import numpy as np
import scipy


def stationary_iterative_solver (
A: scipy.sparse.linalg.LinearOperator ,
b: np.ndarray ,
M_inv: scipy.sparse.linalg.LinearOperator  ,
rtol: float = 1e-6,
x0: None | np.ndarray = None ,
maxiter : int = 100,
callback=None):
    residuals = []
    if x0 is None:
        x0 = np.zeros(len(b))
    
    xm = x0
    b_norm = np.linalg.norm(b)
    for _ in range(0,maxiter):
        res = b-A@xm
        residual = np.linalg.norm(res)/b_norm 
        residuals.append(residual)
        if residual < rtol:
            break
        xm += M_inv @ res

    return xm, residuals