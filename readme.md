This is a (optionally!) GPU-accelerated black-box fast multipole method. On a RTX 2080, it evaluates a 10m-point community transmission problem in 4s per timestep.

While optimized for 2D problems in the million-point-scale, the code supports problems with any number of dimensions, with arbitrary kernels defined entirely in Python.

**If you are working on epidemiological modelling and are interested in using this library, I am keen to help you. Open an issue, or give me an email at `andyjones.ed@gmail.com`.**

## Demo

**This is just a tech demo, not an epidemiological model**

<p align="center"><img src="pybbfmm/demo/demo.gif"></p>

This demonstrates the acceleration of a community transmission kernel. It simulates 10 million agents, the infected of whom emit a cloud a few kilometres across. This cloud represents agents' random interactions in their community. At each step, the method evaluates all 100tn pairs of interactions to figure out who else is at risk of infection. Each step takes about 4s. Extrapolating from smaller problems, the direct approach would take 4h on the same GPU, or about a week on the CPU.

[Take a look at the code for the demo](pybbfmm/demo/__init__.py), or try it out for yourself [in the cloud with Google Colab](https://colab.research.google.com/drive/1Xsmru2czbfVpzGc1e5IW8BOgLmJXBUmN).

## Installation
For just the solver,
```
pip install --upgrade git+https://github.com/andyljones/pybbfmm
```
For the solver and the demo code,
```
pip install --upgrade git+https://github.com/andyljones/pybbfmm#egg=pybbfmm[demo]
```

## Usage
```python
from aljpy import arrdict
import torch
import pybbfmm

prob = arrdict.arrdict(
    # Specify the locations of the sources
    sources=[[0., 0.]],
    # Specify the charges
    charges=[1.],
    # Specify the locations of the targets
    targets=[[1., 1.]])

# Turn it into torch tensors
prob = prob.map(torch.as_tensor)

# Optional: ship it to the GPU
# prob = prob.cuda()
    
# Define the kernel
prob['kernel'] = lambda a, b: 1/((a - b)**2).sum(-1)

# Solve!
soln = pybbfmm.solve(prob)   # tensor([0.5000])
```

## Notes
* This represents a few weeks worth of work. There is a lot of performance still to wring out of the system. I think memory efficiency could probably be upped 2x-4x, and time efficiency 10x with a month or so of effort.
* The main limitation for large problems is memory. With accuracy turned all the way down to `N=1` Chebyshev node per box, about 22m sources & targets can be fit on the 10GB of a RTX 2080 GPU.
* There are various ways to get improvements in that 22m number - like swapping to using ints instead of longs - but the ultimate, scalable solution likely involves streaming parts of the tree to the GPU as needed.
* While the code supports any number of dimensions, 3 and above dims will be _extremely_ slow. The location of the issue is obvious from profiling, but as 3D problems aren't my priority right now I've left it be. 
* The code works just as well on the CPU, though slower. All that's needed is to drop the `.cuda()` call when forming your problem.
* This is part of a larger project about writing a certain kind of epidemiological models as the sum of an n-body problem, a sparse matrix multiplication, and a finite-state machine.

## Background
This grew out of some exploratory work on replicating [Ferguson et al's non-pharmaceutical intervention report](https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf). I found that the slow part of the [underlying spatiotemporal model](https://static-content.springer.com/esm/art%3A10.1038%2Fnature04795/MediaObjects/41586_2006_BFnature04795_MOESM28_ESM.pdf) was the community transmission step, where each contagious person radiates a cloud of infectiousness. This is in many ways similar to how n-body simulations work, and yet I couldn't find anything in the epidemiological literature about accelerating community transmission calculations using fast multipole methods.

I suspect this is because fast multipole methods are fairly tricky to implement, and at least using the traditional approach require a lot of careful analytical expansions. More recent research has introduced [black box fast multipole methods](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf) which let you accelerate n-body-esque simulations while excusing you from doing any hard math.

## Alternatives
There are a [couple](https://github.com/sivaramambikasaran/BBFMM2D) of [Python](https://github.com/DrFahdSiddiqui/bbFMM2D-Python) implementations [around](https://github.com/ruoxi-wang/PBBFMM3D), but none of them are easy to use or modify.
