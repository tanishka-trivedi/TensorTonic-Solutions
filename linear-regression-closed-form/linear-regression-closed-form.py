import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    
    # Write code here
    w = np.linalg.pinv(X) @ y
    return w
    #pass