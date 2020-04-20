This is a prototype optionally-GPU-accelerated black-box fast multipole method. On a RTX 2080, it evaluates a 10m-point community transmission problem in 4s per timestep.

While optimized for 2D problems in the million-point-scale, the code supports problems with any number of dimensions, with arbitrary kernels defined entirely in Python.

## Demo

**This is just a tech demo, not an epidemiological model**

<p align="center"><img src="pybbfmm/demo/demo.gif"></p>

## Background
This grew out of some exploratory work on replicating epidemiological models. I found that the slow part of the spatiotemporal model I was looking at was the community transmission step, where each contagious person radiates a cloud of infectiousness. This is in many ways similar to how n-body simulations work, and yet I couldn't find anything in the epidemiological literature about accelerating community transmission calculations using fast multipole methods.

I suspect this is because fast multipole methods are fairly tricky to implement, and at least using the traditional approach require a lot of careful analytical expansions. More recent research has introduced [black box fast multipole methods](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf) which let you accelerate n-body-esque simulations while excusing you from doing any analytical work.

## Alternatives
There are a [couple](https://github.com/sivaramambikasaran/BBFMM2D) of [Python](https://github.com/DrFahdSiddiqui/bbFMM2D-Python) implementations [around](https://github.com/ruoxi-wang/PBBFMM3D), but none of them are easy to use or modify, and none of them properly leverage the GPU. When complete, I hope this package will change that. 
