# pybbfmm blogpost

`infection-animation.png`

**This is not an epidemiological model**. It's a tech demo for a part of an epidemiological model. What you're looking at is ten million simulated people in the United Kingdom. Each person has a small chance of infecting another person within a few kilometres, and that chance changes with distance.

With ten million people, you'd think that each frame of the above animation would mean you have to calculate a hundred trillion (ten million squared) interactions. Even for modern silicon, that's a lot! But there is a lesser-known algorithm from the 1980s that can do it in a few seconds.

That algorithm - the _fast multipole method_ - is pretty involved in its entirety. But it has a key idea in it that's widely applicable, and for its sake I'm going to explain the algorithm without using any maths at all.

>  **The Bit For People With CS Degrees**
> Fast multipole methods turn quadratic-time interaction problems into linear-time problems, and a recent version called the black-box multipole method can do it for any interaction you choose. The key idea is _a recursive sense of distance_, and that's what most of this post is aimed towards explaining.

## Setup

Here's a _source_ and its _field_:

`source-and-field`

Maybe the source is a planet and the field is its gravity. Maybe the source is a particle and the field is the electric field. Maybe the source is an infected person and the field is the transmission risk. 

Here're hundred sources and their combined fields:

`sources-and-fields`

To draw that plot, we have to calculate the combined field at 100 different points. That means

  * take a point
  * take each of the hundred sources in turn
  * calculate the contribution that source makes to the field at that point
  * add all the contributions up
  * repeat 99 times for the 99 other points 

for a total of 10,000 contribution calculations.

That's pretty wasteful when you think about it. I mean, these two sources make about the same contribution to this point

`two-sources-contribution-one-point`

So we're basically doing the same calculation twice! Can we avoid that? 

## Cells

How about this instead: divide the world up into 32 cells, and gather all the sources in each cell to the middle of the cell: 

`left-point-right-cell`

For example, here we're calculating the field at this point on the left. By gathering all the sources in the cell on the right to the center of the cell, we can just calculate one contribution - between the column and the cell center - and then multiply the answer by the number of points! It works pretty well: the approximated field is almost identical to the actual field.

But, ah, what about this cell that's right on top of the point we're trying to calculate the field for:

`left-point-left-cell`

This time the approximation is _really_ different to the actual field. Rats.

See, the problem is this: when a source is a long way away from a pixel, it doesn't really matter exactly where the source is. The field is pretty flat when you're far away! But up close, it matters a lot where the source is: the field is very steep there.

`shallow-field-steep-field`

OK, so, new rule: 

  * You can calculate contributions from distant cells by gathering their sources
  * You have to calculate contributions from nearby sources one-by-one

What's 'distant' and 'nearby' here? Well, the worst-case scenario is when a pixel is right next to a source. Then the field is really steep and we definitely want to calculate that contribution exactly. A source can be right next to a point if they are in the same cell, or if they're in neighbouring cells:

`point-close-to-source`

So a more explicit rule is:

  * You can calculate contributions from sources in cells more than one cell away from the point by gathering their sources
  * You have to calculate contributions from sources in the same cell or neighbouring cells one-by-one

Trying it out, it's pretty good:

`distant-cell-approx`

In this particular example, rather than the original 10k contributions we're now calculating 2900 contributions from distant cells, and 900 contributions from nearby sources. That's a total of 3800 contributions to calculate, down 60% from the original approach!

Can we do better?

## Bigger cells
Well, the cell sizes we - I - chose above were totally arbitrary. There's no reason they can't be twice as big:

`cell-groups-twice`

Now there are 1300 contributions from distant cells - which is better than before! - and 1800 contributions from nearby sources - which is worse than before! By making the cells bigger, there are now fewer distant cells to calculate contributions from, but more neighbouring sources whose contributions have to be calculated one-by-one. 

But there is a way to get the best of both worlds. Look at both the bigger and smaller cells _together_:

`bigger-smaller-cells`

Most of the bigger cells are far from the point we're looking at, so we can gather the sources together in each of those distant big cells just fine. The trouble is the three big cells near to the point. But there, we can look at the smaller cells instead! Of the smaller cells beneath those three nearby big cells, three are far away enough that we can gather the sources there, and three are close enough that we have to count sources exactly. 

This is the first key idea in the fast multipole method: **the further away a source is, the less accurate you need to be about _exactly_ where it is**. This is because the further away as source is, the flatter the field is at the point you're looking at.

We can repeat the double-the-size-of-cells thing a few more times:

`bigger-cell-repeated`

Now for each point there's one distant cell at the top layer, and three _newly_ distant cells in each lower layer. In total, there are 700 contributions from distant sources, along with the 900 contributions from nearer ones. 1600 total makes for a 85% improvement over the original approach!

Can we do better?

## Turning it around
The key idea above is that the further away a source is, the less accurate you have to be about exactly where it is. But we can rephrase that and say: the further away a _point_ is, the less accurate you have to be about where it is! 

Let's put our hundred points into the same 32 cells as we did our sources:

`points-in-cells`

All the points in this cell on the left get roughly the same contribution from the sources in this distant cell on the right:

`left-cell-right-cell`

So we can calculate the contribution _from_ all the sources in the cell on the left _to_ all the points in the cell on the right in _with one calculation_. Even better, we can do the same bigger-cells trick as before and get this:

`turned-around-bigger-cells-repeated`

There are now 200 contributions from distant sources and 900 from nearby ones. 1100 total makes for a 90% improvement over the original approach!

## An algorithm
And that's the core of the fast multipole method: recursive approximation, increasing in coarseness as distance increases. Putting it in code, it looks like this:

  * Build a binary tree of cells over your sources and points

`tree-building`

  * Count how many sources are in each cell at the bottom

`source-counting`

  * Sweep up through the tree, adding together the source counts from each pair of children cells to get the number of sources in the parent cell   

`sweep-up`

  * Sweep _down_ through the tree, at each layer looking at pairs of newly-distant cells and calculating the contribution from the sources in one cell to the points in the other.

`sweep-down`

  * At the bottom, add to each point the contribution from the sources in neighbouring cells.

`neighbouring-pairs`

And with that, a quadratic problem is turned into a linear one.

That's the end of the explanatory part of this post. The stuff below is about how the method is used in the real world in its full generality, and about how to go about implementing it yourself.

## Real-world problems
The problem explained above is simplified a _lot_ from the kind you'd see in the wild. 

The first - obvious - simplification is that the problems given here are all 1D, when the typical problem of interest is 2D or 3D. The same ideas work just as well in higher dimensions, and it's possible to design code that handles all of them simultaneously.
    
Next, the sources here all make the same strength contribution. In the wild, each source usually has a differing mass or charge or infectiousness. This is easy enough to handle: rather than counting the sources when you gather the sources to the middle of a cell, you sum them instead. 

Finally, the sources and points in the problem above are evenly distributed. Each cell at the bottom has roughly the same number of sources and points in, while in the real world this isn't usually the case. The fix is to replace the 'full' binary tree shown above with an 'adaptive' binary tree that splits further in regions of higher density. This introduces a fairly substantial amount of complexity.

In its full generality, the fast multipole method will accelerate any problem that can be written like this
$$
\text{field at point } i = \sum_j (\text{contribution to point } i \text{ from source } j)   \cdot \text{strength of source } j
$$
where the contribution function is 'nice' in some reasonable ways, most often that it's increasingly flat as you move away from the source.

## Real-world approximations
In the algorithm described above, the approximation used is the simplest possible: it's a constant across the cell, and the constant is the field strength at the center of the cell. This has the advantage of being easy to explain, and frankly for many purposes it'll do just fine. My experiments with using a constant approximation give an RMSE of about .1%. But if you want higher accuracy than that, either you need to widen the neighbourhood you consider 'near' so more source-point pairs get their contributions calculated exactly, or you need a better approximation. 

When the algorithm was first developed for physics simulations in the 1980s, the approximation of choice was a Laurent series. It makes for a really good approximation, but requires a fair bit of hand-rolled maths catered to exactly the domain you're looking at. This is where the name 'multipole method' comes from, as Laurent series are defined around 'poles' where the approximation becomes infinite. The name is a bit of a shame, since the key idea has nothing to do with Laurent series or poles. I suspect if it'd been called the 'recursive approximation method', it'd be a lot more widely known.

In the more recent 'black-box' version from the late 2000s, instead of a Laurent approximation a Chebyshev approximation is used. The advantage of the Chebyshev approximation is that it requries no domain-specific tuning at all: you can pass an arbitrary kernel to the code and it'll figure things out by evaluating it at a handful of points. The downside to the Chebyshev approximation is that it requires more variables and more compute for a given level of accuracy than the Laurent approach does, though there are (again) more complex ways to fix this.

In both cases, the approximation can have its accuracy improved at the expense of being slower. 

`approximations`

## Real-world implementations
TODO: this bit
  * Tree construction when your points aren't sorted
  * Dynamic trees
  * Parallelisation
  * Papers
  * Reference implementations