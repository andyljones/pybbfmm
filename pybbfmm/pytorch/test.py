import torch
from aljpy import arrdict
import numpy as np

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(S=3, T=5, D=2):
    return arrdict.arrdict(
        sources=np.random.uniform(-1., +1., (S, D)),
        charges=np.random.uniform(.1, 1., (S,)),
        targets=np.random.uniform(-1., +1., (T, D))).map(torch.tensor).float().cuda()

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