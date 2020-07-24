import torch

def cartesian_product(xs, D):
    """Returns all D-tuples of ``xs`` as a (n_xs, n_xs, ..., \\*tail_dims)-tensor.
    """
    xs = torch.as_tensor(xs)
    return torch.stack(torch.meshgrid(*((xs,)*D)), -1)

def flat_cartesian_product(xs, D):
    """Returns all D-tuples of ``xs`` as a (n_xs :sup:`D`, \\*tail_dims)-tensor.
    """
    return cartesian_product(xs, D).reshape(-1, D)