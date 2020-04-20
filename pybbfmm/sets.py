import torch

def cartesian_product(xs, D):
    xs = torch.as_tensor(xs)
    return torch.stack(torch.meshgrid(*((xs,)*D)), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)