import torch

class Ragged:

    def __init__(self, image, cardinalities):
        """A ragged tensor, which is to a tensor of different-length tensors.

        The different-length sub-tensors are concatenated together to make a single backing tensor. This makes
        for fast element-wise operations.

        A lot of the language in this class's implementation comes from thinking of a ragged tensor as a mapping
        from indices to sub-tensors. 

        :param image: the sub-tensors, concatenated together in order. The image of the mapping.
        :param cardinalities: the number of elements in each sub-tensor.
        """
        self._image = torch.as_tensor(image)
        self._cardinalities = torch.as_tensor(cardinalities)
        self._starts = self._cardinalities.cumsum(0) - self._cardinalities
        self.max_k = self._cardinalities.max()

    @property
    def domain(self):
        """Returns the size of the domain - ie the number of sub-tensors."""
        return len(self._cardinalities)

    @property
    def range(self):
        """Returns the size of the arnge - ie the sum of the lengths of all the subtensors, or equivalently the size
        of the backing tensor."""
        return len(self._image)

    def kth(self, qs, c):
        """For each ``q`` in ``qs``, returns the ``c`` th element of the ``q`` th subtensor.

        Only returns valid values - ie it only returns the results for which ``q`` is at least ``c+1`` elements long.
        To indicate which ``q`` were valid, it also returns a mask. 

        For example, ::

            # Sub-tensors are [1, 2], [10, 11, 12], [20] 
            r = Ragged([1, 2, 10, 11, 12, 20], [2, 3, 1])
            r.kth([1, 2], 1)
            # [11], [True, False]

        One way to think about this is as if the sub-tensors were arranged in a grid

        .. code:: text

             
            q c 0   1   2
            0  [1,  2]
            1  [10, 11, 12]
            2  [20]

        Calling ``r.kth([1, 2], 1)`` means taking rows [1, 2] and column 1. We get back the values that exist in this 
        column - ``[11]`` and a mask saying which rows stretch as far as column ``c``.
        """ 
        qs = torch.as_tensor(qs)
        valid = self._cardinalities[qs] > c
        indices = self._starts[qs[valid]] + c
        return self._image[indices], valid

    def slice(self, idx):
        """Returns the slice corresponding to the ``idx`` th sub-tensor."""
        return slice(self._starts[idx], self._starts[idx]+self._cardinalities[idx])

    def image(self, idx):
        """Returns the ``idx`` th sub-tensor."""
        return self._image[self.slice(idx)]

    def __repr__(self):
        return f'{type(self).__name__}({len(self._starts)}, {len(self._image)})'

    def __str__(self):
        return repr(self)

def from_pairs(pairs, n_ps, n_qs):
    """Creates a :class:`Ragged` from a list of index-value pairs. A single index can occur many times."""
    qs, ps = pairs.T
    sort = torch.sort(qs)
    image = ps[sort.indices]
    # As of Pytorch 1.4, unique_consecutive breaks on empty inputs. 
    if len(pairs) > 0:
        unique, counts = torch.unique_consecutive(sort.values, return_counts=True)
    else:
        unique, counts = torch.zeros_like(sort.values), torch.zeros_like(sort.values)


    cardinalities = qs.new_zeros(n_qs)
    cardinalities[unique] = counts 

    return Ragged(image, cardinalities)

def invert_indices(qs, n_qs):
    """Creates a :class:`Ragged` from a list of values, by interpreting the list as a index-to-value mapping.
    
    The ragged is then the mapping from value-to-indices-with-that-value."""
    sort = torch.sort(qs)
    # As of Pytorch 1.4, unique_consecutive breaks on empty inputs. 
    if len(qs) > 0:
        unique, counts = torch.unique_consecutive(sort.values, return_counts=True)
    else:
        unique, counts = torch.zeros_like(sort.values), torch.zeros_like(sort.values)

    cardinalities = qs.new_zeros(n_qs)
    cardinalities[unique] = counts 
    return Ragged(sort.indices, cardinalities)

def test_invert():
    qs = torch.tensor([5, 3, 3, 2, 0])
    ps = invert_indices(qs, max(qs)+1)

    im, ma = ps.kth(qs, 0)

    torch.testing.assert_allclose(qs, qs[im])