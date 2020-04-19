# p_to_q: indexing with a `p` returns one `q`
# q_to_p: indexing with a `q` could return several `p`. 
# We want to vectorize the indexing, so need to specify a cardinal too.  
# Since some `q`s in the vector might not have a matching cardinal, need to provide
# a way to mask a vector of `q`s down to the valid ones for that cardinality.
#
# So:
#  * Store in (p, cardinality) order
#  * Store p's cardinality
#  * Store where p's bit of storage begins
#  * When queried with a p and a multiplicity, mask out to valid ps, then return the 
#    mask and the image 
import torch

class Ragged:

    def __init__(self, image, cardinalities):
        self._image = image
        self._cardinalities = cardinalities
        self._starts = cardinalities.cumsum(0) - cardinalities
        self.max_k = cardinalities.max()

    @property
    def domain(self):
        return len(self._cardinalities)

    @property
    def range(self):
        return len(self._image)

    def kth(self, qs, c):
        qs = torch.as_tensor(qs)
        valid = self._cardinalities[qs] > c
        indices = self._starts[qs[valid]] + c
        return self._image[indices], valid

    def slice(self, idx):
        return slice(self._starts[idx], self._starts[idx]+self._cardinalities[idx])

    def image(self, idx):
        return self._image[self.slice(idx)]

    def __repr__(self):
        return f'{type(self).__name__}({len(self._starts)}, {len(self._image)})'

    def __str__(self):
        return repr(self)

def from_pairs(pairs, n_ps, n_qs):
    qs, ps = pairs.T
    sort = torch.sort(qs)
    image = ps[sort.indices]
    unique, counts = torch.unique_consecutive(sort.values, return_counts=True)

    cardinalities = qs.new_zeros(n_qs)
    cardinalities[unique] = counts 

    return Ragged(image, cardinalities)

def invert_indices(qs, n_qs):
    sort = torch.sort(qs)
    unique, counts = torch.unique_consecutive(sort.values, return_counts=True)
    cardinalities = qs.new_zeros(n_qs)
    cardinalities[unique] = counts 
    return Ragged(sort.indices, cardinalities)

def test_invert():
    qs = torch.tensor([5, 3, 3, 2, 0])
    ps = invert_indices(qs, max(qs)+1)

    im, ma = ps.kth(qs, 0)

    torch.testing.assert_allclose(qs, qs[im])