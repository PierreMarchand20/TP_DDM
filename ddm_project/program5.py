import scipy.sparse as sp
import numpy as np


def global_matrix_vector_product (A:list[np.ndarray],x: list[np.ndarray], D:list[np.ndarray],neighbors:dict[int, list[int]], intersections: dict[int, list[list[int]]],exchange_indices:list[list[int]],ovr_subdomain_to_global:dict[int, np.ndarray],dofs_global: int | None) -> np. ndarray:
    """Compute global matrix_vector_product

    Args:
        A (list[np.ndarray]): Local matrices
        x (list[np.ndarray]): Local vectors
        D (list[np.ndarray]): Partition of unity

    Returns:
        list[ np. ndarray]: global matrix vector product

    """
    nb_partition = len(x)
    y_own = []
    y = []
    # Compute the own contributions
    for i in range(0,nb_partition):
        y_p = A[i]@(D[i]*x[i]) 
        y.append(y_p)
        y_own.append(y_p)
    
    
    for i in range(0,nb_partition):
        for j,n in enumerate(neighbors[i]):
            y[i][intersections[i][j]] += y_own[n][exchange_indices[i][n]]
    
    if dofs_global is not None:
        y_global = np.zeros(dofs_global)
        for i in range(0,nb_partition):
            y_global[ovr_subdomain_to_global[i]] = y[i]

    return y_global
        