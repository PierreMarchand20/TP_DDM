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
    y_global = np.zeros_like(x)

    # Compute the own contributions
    for i in range(0,nb_partition):
        x_p = x[ovr_subdomain_to_global[i]]
        y_global[ovr_subdomain_to_global[i]] += A[i]@(D[i]*x_p) 
        

    return y_global
        

def apply_ASM_precondition(Aps_inv: list[scipy.sparse.linalg.SuperLU ],x: np.ndarray,ovr_subdomain_to_global:dict[int, np.ndarray],nb_partition : int,boundary_nodes: list[np.ndarray]) -> np.ndarray:
    """Implements the ASM preconditioner

    Args:
        input (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    y = np.zeros_like(x)

    for i in range(0,nb_partition):
        x_p = x[ovr_subdomain_to_global[i]]
        x_p[boundary_nodes[i]] = 0
        y_p = Aps_inv[i].solve(x_p)
        y[ovr_subdomain_to_global[i]] += y_p

    return y
    

def apply_RAS_precondition(Aps_inv: list[scipy.sparse.linalg.SuperLU ],x: np.ndarray,ovr_subdomain_to_global:dict[int, np.ndarray],nb_partition : int,boundary_nodes: list[np.ndarray],D: list[np.ndarray]) -> np.ndarray:
    """Implements the ASM preconditioner

    Args:
        input (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    y = np.zeros_like(x)
    # y= x
    for i in range(0,nb_partition):
        x_p = x[ovr_subdomain_to_global[i]]
        x_p[boundary_nodes[i]] = 0
        # print(f"x[{i}] = {x_p}")
        y_p = Aps_inv[i].solve(x_p)
        # y_p = Aps_inv[i].solve(D[i]*x_p)
        # print(f"D[{i}] = {D[i]}")
        y[ovr_subdomain_to_global[i]] += D[i]*y_p

    return y




if __name__ == "__main__":
    print("Running tests")

    A1 = np.ndarray([[0,0,1],[1 , 0 ,0],[0,1,0]])
    A2 = np.ndarray([[0,0,-1],[-1 , 0 ,0],[0,-1,0]])