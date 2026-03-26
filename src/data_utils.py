import torch 

EPS = 1e-6

def normalize_instance(x, obs_mask):
    """
    x: 1D tensor of shape (T,)
    obs_mask: bool tensor of shape (T,)
    """

    observed = x[obs_mask]
    mu = observed.mean()
    sigma = observed.std()
    x_norm = (x-mu)/(sigma+EPS)

    return x_norm