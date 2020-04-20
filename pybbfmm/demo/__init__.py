import pickle
import pathlib
import torch
from aljpy import arrdict
import numpy as np
from tqdm.auto import tqdm
from .. import presolve, evaluate
from . import population, plotting

def risk_kernel(a, b):
    # Community transmission kernel from `Strategies for mitigating an influenza pandemic.`,
    # multiplied by a hand-picked constant to get the animation looking nice.
    d = (a - b).pow(2).sum(-1).pow(.5)
    return .0001 * 1/(1 + (d/4)**3)

def adapt(risk):
    # Infection risk is multiplicative in its complement: if you're exposed
    # to two infected people, your chance of infection is $1 - (1 - r)^2$.
    # 
    # n-body solvers expect additive kernels though, so here we take the
    # log of the compliment. 
    def log_nonrisk(a, b):
        return torch.log(1 - risk(a, b).clamp(None, .9999))

    return log_nonrisk

def simulate(n=10e6, T=120):
    # Get a population
    pop = population.points(n=n)

    # Phrase it as an n-body problem and stick it on the GPU
    prob = arrdict.arrdict(
        sources=pop,
        targets=pop,
        charges=np.zeros(len(pop))
    ).map(torch.as_tensor).float().cuda()

    # Wrap the risk kernel so it can be fed into the solver
    prob['kernel'] = adapt(risk_kernel)

    # Do the presolve
    presoln = presolve(prob)

    # Set patient zero - the infection status is independent of the presolve!
    presoln.scaled.charges[0] = 1.

    # Run the simulation!
    infected = [presoln.scaled.charges.cpu().numpy()]
    for t in tqdm(range(T)):
        # Evaluate the total risk over all n^2 pairs of points
        log_nonrisk = evaluate(**presoln)
        risk = 1 - torch.exp(log_nonrisk)
        
        # Figure out which people got infected
        rands = torch.rand_like(risk)
        presoln.scaled.charges = ((rands < risk) | (0 < presoln.scaled.charges)).float()

        # Add the last step's result to the output
        infected.append(presoln.scaled.charges.cpu().numpy())
    
    return infected, prob.targets.cpu().numpy()

def save(infected, points):
    import pathlib
    import pickle

    pkl = pickle.dumps({'points': points, 'infected': infected})
    pathlib.Path('output/tmp.pkl').write_bytes(pkl)

def load():
    return pickle.loads(pathlib.Path('output/tmp.pkl').read_bytes())

def run():
    infected, points = simulate()
    save(infected, points)
    plotting.animate(infected, points)