import numpy as np

def normalize_3d(v):
    v = np.array(v, dtype=float)

    norms = np.linalg.norm(v, axis=-1, keepdims=True)

    # Avoid division by zero (leave zero vectors unchanged)
    norms[norms == 0] = 1

    return v / norms