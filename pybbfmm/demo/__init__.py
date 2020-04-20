import pickle
import pathlib
import torch
from aljpy import arrdict
import numpy as np
from tqdm.auto import tqdm
from .. import presolve, evaluate
from . import population, plotting

def scale_risk_kernel(n):

    # For demo purposes, we want to scale infectiousness against pop 
    infectiousness = 1e-4/(n/10e6)

    def risk_kernel(a, b):
        # Community transmission kernel from `Strategies for mitigating an influenza pandemic.`,
        # multiplied by a hand-picked constant to get the animation looking nice.
        d = (a - b).pow(2).sum(-1).pow(.5)
        return infectiousness * 1/(1 + (d/4)**3)
    
    return risk_kernel

def wrap(risk):
    # Infection risk is multiplicative in its complement: if you're exposed
    # to two infected people, your chance of infection is $1 - (1 - r)^2$.
    # 
    # n-body solvers expect additive kernels though, so here we take the
    # log of the compliment. We'll recover the infection risk after the 
    # solver's done the summing for us.
    def log_nonrisk(a, b):
        return torch.log(1 - risk(a, b).clamp(None, .9999))

    return log_nonrisk

def simulate(n=10e3, T=40, device='cpu'):
    print(f'This demo will be for {int(n)} agents and {T} timesteps on device "{device}".')
    print('The default values are fairly small, so as not to frustrate anyone with out-of-memory errors. Pass larger ones if you want.')
    print('Pass device="cuda" to run on the GPU')

    # Get a population
    pop = population.points(n=n)

    # Phrase it as an n-body problem and stick it on the GPU
    prob = arrdict.arrdict(
        sources=pop,
        targets=pop,
        charges=np.zeros(len(pop))
    ).map(torch.as_tensor).float().to(device)

    # Create our risk kernel. Usually this'd be hard coded, but we want the
    # demo to be interesting for a range of population densities, so it needs
    # to be variable.
    risk_kernel = scale_risk_kernel(n)

    # Wrap the risk kernel so it can be fed into the solver
    prob['kernel'] = wrap(risk_kernel)

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
    pathlib.Path('output').mkdir(exist_ok=True, parents=True)
    pathlib.Path('output/tmp.pkl').write_bytes(pkl)

def load():
    return pickle.loads(pathlib.Path('output/tmp.pkl').read_bytes())

def run(**kwargs):
    print('Running simulation...')
    infected, points = simulate(**kwargs)
    save(infected, points)
    print('Rendering...')
    plotting.animate(infected, points)