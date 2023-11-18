import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    rng = np.random.default_rng()
    r = rng.standard_normal(data.shape[1])
    for _ in range(num_steps):
        r = data.dot(r)
        r = r / np.linalg.norm(r)
    return float(r.T.dot(data.dot(r)) / r.T.dot(r)), r