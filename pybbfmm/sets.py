import torch

def cartesian_product(xs, D):
    xs = torch.as_tensor(xs)
    return torch.stack(torch.meshgrid(*((xs,)*D)), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)

def left_index(A):
    return torch.stack([torch.arange(len(A), dtype=A.dtype, device=A.device), A], -1)

def right_index(A):
    return torch.stack([A, torch.arange(len(A), dtype=A.dtype, device=A.device)], -1)

def inner_join(A, B):
    """Given an array of pairs (a, p) and another of pairs (q, b), returns all pairs (a, c)
    such that for there's some pair (a, r) in the first array and another (r, b) in the second.
    
    Which is to say, it's an inner join. 
    """
    if (len(A) == 0) or (len(B) == 0):
        return A[:0]

    A_order = torch.argsort(A[:, 1])
    A_sorted = A[A_order]
    A_unique, A_inv, A_counts = torch.unique(A_sorted[:, 1], return_inverse=True, return_counts=True)

    B_order = torch.argsort(B[:, 0])
    B_sorted = B[B_order]
    B_unique, B_inv, B_counts = torch.unique(B_sorted[:, 0], return_inverse=True, return_counts=True)

    C_unique, C_inv = torch.unique(torch.cat([A_unique, B_unique]), return_inverse=True)
    A_unique_inv, B_unique_inv = C_inv[:len(A_unique)], C_inv[len(A_unique):]

    CA_counts = torch.zeros_like(C_unique)
    CA_counts[A_unique_inv] = A_counts

    CB_counts = torch.zeros_like(C_unique)
    CB_counts[B_unique_inv] = B_counts

    pairs = []
    for A_reps in range(1, CA_counts.max()+1):
        for B_reps in range(1, CB_counts.max()+1):
            mask = (CA_counts == A_reps) & (CB_counts == B_reps)
            
            A_vals = A_sorted[mask[A_unique_inv[A_inv]], 0]
            B_vals = B_sorted[mask[B_unique_inv[B_inv]], 1]

            pairs.append(torch.stack([
                A_vals.reshape(mask.sum(), A_reps, 1).repeat_interleave(B_reps, 2).flatten(),
                B_vals.reshape(mask.sum(), 1, B_reps).repeat_interleave(A_reps, 1).flatten()], -1))
    pairs = torch.cat(pairs)

    return pairs

def accumulate(indices, vals, length):
    totals = vals.new_zeros((length,) + vals.shape[1:])
    totals.index_add_(0, indices, vals)
    return totals

def as_linear_index(subscripts):
    breadth = subscripts.max(0).values + 1
    bases = breadth.cumprod(0).div(breadth).flip((0,))
    return (subscripts*bases).sum(-1), bases

def as_subscripts(linear, bases):
    coords = []
    for b in bases[:-1]:
        coords.append(linear // b)
        linear = linear - b*coords[-1]
    return torch.stack(coords + [linear], -1)

def unique_rows(rows):
    if len(rows) == 0:
        return rows, rows.new_empty((0,))
    mins = rows.min(0).values
    subscripts = rows - mins
    linear, bases = as_linear_index(subscripts)

    unique, inv = torch.unique(linear, return_inverse=True)
    unique = as_subscripts(unique, bases) + mins
    return unique, inv