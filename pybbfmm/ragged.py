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

    def __init__(self, image, cards):
        self._image = image
        self._cards = cards
        self._starts = cards.cumsum(0)

    def __getitem__(self, ps, c):
        valid = self._cards[ps] > c
        indices = self._starts[ps[valid]] + c
        return self._image[indices], valid

def invert(p_to_q, length):
    pass


def test_invert():
    ps = torch.tensor([5, 3, 3, 2, 0])
    qs = invert(ps)

    im, ma = qs[ps, 0]