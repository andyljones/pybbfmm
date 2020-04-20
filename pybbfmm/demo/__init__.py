import torch
from aljpy import arrdict
import numpy as np
from tqdm.auto import tqdm
from .. import presolve, evaluate
from . import population

def risk_kernel(a, b):
    d = (a - b).pow(2).sum(-1).pow(.5)
    return .0001 * 1/(1 + (d/4)**3)

def kernel(a, b):
    # We want to take products of non-infection kernels here
    # Cutoff at .9999 risk to suppress infinities
    return torch.log(1 - risk_kernel(a, b).clamp(None, .9999))

def nbody_problem(pop):
    prob = arrdict.arrdict(
        sources=pop,
        targets=pop,
        charges=np.zeros(len(pop))
    ).map(torch.as_tensor).float().cuda()
    prob['kernel'] = kernel
    return prob

def simulate(n=10e3):
    # Get a population
    pop = population.points(n=n)

    # Phrase it as an n-body problem and do the pre-solve.
    prob = nbody_problem(pop)
    presoln = presolve(prob)

    # Set patient zero
    presoln.scaled.charges[0] = 1.

    infected = []
    for t in tqdm(range(120)):
        infected.append(presoln.scaled.charges.cpu().numpy())

        log_nonrisk = evaluate(**presoln)
        risk = 1 - torch.exp(log_nonrisk)
        
        rands = torch.rand_like(risk)
        presoln.scaled.charges = ((rands < risk) | (0 < presoln.scaled.charges)).float()
    
    return infected, prob.targets.cpu().numpy()

def run():
    infected, points = simulate()
    plotting.animate(infects, points)