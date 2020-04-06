from scipy.signal.signaltools import asarray, array, _init_nd_shape_and_axes_sorted, \
    np, _inputs_swap_needed, fftpack, _rfft_lock, _rfft_mt_safe, _centered

def fftconvolve(in1, in2, mode="full", axes=None):
    """This is identical to scipy fftconvolve in scipy 1.3, except disabling
    the valid-dims check"""
    in1 = asarray(in1)
    in2 = asarray(in2)
    noaxes = axes is None

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])

    _, axes = _init_nd_shape_and_axes_sorted(in1, shape=None, axes=axes)

    if not noaxes and not axes.size:
        raise ValueError("when provided, axes cannot be empty")

    if noaxes:
        other_axes = array([], dtype=np.intc)
    else:
        other_axes = np.setdiff1d(np.arange(in1.ndim), axes)

    s1 = array(in1.shape)
    s2 = array(in2.shape)

    if not np.all((s1[other_axes] == s2[other_axes])
                  | (s1[other_axes] == 1) | (s2[other_axes] == 1)):
        raise ValueError("incompatible shapes for in1 and in2:"
                         " {0} and {1}".format(in1.shape, in2.shape))

    complex_result = (np.issubdtype(in1.dtype, np.complexfloating)
                      or np.issubdtype(in2.dtype, np.complexfloating))
    shape = np.maximum(s1, s2)
    shape[axes] = s1[axes] + s2[axes] - 1

    #TODO: Disabled check.
    # # Check that input sizes are compatible with 'valid' mode
    # if _inputs_swap_needed(mode, s1, s2):
    #     # Convolution is commutative; order doesn't have any effect on output
    #     in1, s1, in2, s2 = in2, s2, in1, s1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [fftpack.helper.next_fast_len(d) for d in shape[axes]]
    fslice = tuple([slice(sz) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.
    if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
        try:
            sp1 = np.fft.rfftn(in1, fshape, axes=axes)
            sp2 = np.fft.rfftn(in2, fshape, axes=axes)
            ret = np.fft.irfftn(sp1 * sp2, fshape, axes=axes)[fslice].copy()
        finally:
            if not _rfft_mt_safe:
                _rfft_lock.release()
    else:
        # If we're here, it's either because we need a complex result, or we
        # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
        # is already in use by another thread).  In either case, use the
        # (threadsafe but slower) SciPy complex-FFT routines instead.
        sp1 = fftpack.fftn(in1, fshape, axes=axes)
        sp2 = fftpack.fftn(in2, fshape, axes=axes)
        ret = fftpack.ifftn(sp1 * sp2, axes=axes)[fslice].copy()
        if not complex_result:
            ret = ret.real

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_valid = shape.copy()
        shape_valid[axes] = s1[axes] - s2[axes] + 1
        return _centered(ret, shape_valid)
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")