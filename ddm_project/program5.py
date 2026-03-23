import scipy.sparse as sp
import numpy as np
import scipy

def global_matrix_vector_product (A:list[np.ndarray],x: np.ndarray, D:list[np.ndarray],ovr_subdomain_to_global:dict[int, np.ndarray]) -> np. ndarray:
    """Compute global matrix_vector_product

    Args:
        A (list[np.ndarray]): Local matrices
        x np.ndarray: Global vector
        D (list[np.ndarray]): Partition of unity

    Returns:
        list[ np. ndarray]: global matrix vector product

    """
    nb_partition = len(A)
    y_own = []
    y = []
    y_global = np.zeros_like(x)

    # Compute the own contributions
    for i in range(0,nb_partition):
        print(i)
        x_p = x[ovr_subdomain_to_global[i]]
        y_global[ovr_subdomain_to_global[i]] += A[i]@(D[i]*x_p) 
        

    return y_global
        

def apply_ASM_precondition(As: list[scipy.sparse.linalg.SuperLU ],x: np.ndarray,ovr_subdomain_to_global:dict[int, np.ndarray],nb_partition : int) -> np.ndarray:
    """Implements the ASM preconditioner

    Args:
        input (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    y = np.zeros_like(x)

    for i in range(0,nb_partition):
        x_p = x[ovr_subdomain_to_global[i]]
        y_p = As[i].solve(x_p)
        y[ovr_subdomain_to_global[i]] += y_p

    return y
    

if __name__ == "__main__":
    print("Running tests")