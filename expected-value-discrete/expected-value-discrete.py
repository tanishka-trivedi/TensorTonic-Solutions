import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)
    
    # Hint 1: validate probabilities
    if not np.allclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")
    
    # Hint 2: compute expected value
    return np.sum(x * p)
