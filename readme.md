This grew out of some exploratory work on replicating epidemiological models. I found that the slow part of the spatiotemporal model I was looking at was the community transmission step, where each contagious person radiates a cloud of infectiousness. This is in many ways similar to how n-body simulations work, and yet I couldn't find anything in the epidemiological literature about accelerating community transmission calculations using fast multipole methods.

I suspect this is because fast multipole methods are fairly tricky to implement, and at least using the traditional approach require a lot of careful analytical expansions. More recent research has introduced [black box fast multipole methods](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf) which let you accelerate n-body-esque simulations while excusing you from doing any analytical work.

While there are [several](https://github.com/sivaramambikasaran/BBFMM2D) [Python](https://github.com/DrFahdSiddiqui/bbFMM2D-Python) [implementations](https://github.com/ruoxi-wang/PBBFMM3D) around, none of them are easy to use or modify. When complete, I hope this package will change that. 

## Notes
Install JAX with
```
pip install --upgrade git+https://github.com/hawkinsp/jax.git@scan
```
Should make it into a JAX release soon. 