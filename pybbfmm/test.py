import scipy as sp
import torch
from aljpy import arrdict
import numpy as np

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(S=3, T=5, D=2, device='cuda'):
    prob = arrdict.arrdict(
        sources=np.random.uniform(-1., +1., (S, D)),
        charges=np.random.uniform(.1, 1., (S,)),
        targets=np.random.uniform(-1., +1., (T, D))
    ).map(lambda t: torch.as_tensor(t).float().to(device))
    prob['kernel'] = quad_kernel
    return prob

def v_w_problem():
    return arrdict.arrdict(
        sources=torch.tensor([
            [-.25, +.75],
            [-.75, +.75]]),
        charges=torch.tensor([1., 1.]),
        targets=torch.tensor([
            [+.25, +.75],
            [+.75, +.75]]),
        scale=torch.tensor([1., 1.])
    ).cuda()


def solve(prob, kernel=quad_kernel):
    k = kernel(prob.targets[:, None], prob.sources[None, :])
    return (k*prob.charges).sum(-1)

REPORT_MEMORY = []
def report_memory(f):

    def g(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = f(*args, **kwargs)
        REPORT_MEMORY.append((
            f.__name__, 
            torch.cuda.max_memory_allocated()/1e6))
        return result
    
    return g

def memory_report():
    import pandas as pd
    global REPORT_MEMORY
    report = REPORT_MEMORY.copy()
    REPORT_MEMORY = []
    return pd.DataFrame(report, columns=['name', 'size'])

def memory_usage(d):
    if isinstance(d, dict):
        return type(d)({k: memory_usage(v) for k, v in d.items()})
    if isinstance(d, list):
        return np.array([memory_usage(x) for x in d])
    if isinstance(d, torch.Tensor):
        return d.nelement()*d.element_size()/1e6

def plot_occupancy(D, cutoff, depths=30):
    qs = {}
    for M in np.logspace(1, np.log10(100e6), 500):
        depths = np.arange(depths)
        occ = 0.
        prev_p = 0.
        for depth in depths:
            cells = 2**(depth*D)
            mu = M/cells
            p_cell_full = 1 - sp.stats.poisson(mu=mu).cdf(cutoff)
            p_one_sufficient = (1 - p_cell_full)**cells
            p_both_sufficient = 1 - (1 - p_one_sufficient)**2
            occ += (p_both_sufficient - prev_p)*mu/cutoff
            prev_p = p_both_sufficient
        qs[M] = occ
    qs = pd.Series(qs)

    ax = qs.plot(logx=True)
    ax.set_xlim(10**3, qs.index[-1])
    ax.set_ylim(0, .25)
    ax.grid(True)
    ax.set_title(f'Occupancy, cutoff={cutoff}, D={D}')