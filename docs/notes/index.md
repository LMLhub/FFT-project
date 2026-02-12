# Notes

This section contains conceptual and theoretical background.

## Preference learning

Predicting the best gamble between two options $(g^{(1)}, g^{(2)})$ is not
exactly equivalent to standard binary classification because the preferred
choice should not depend on the order in which the options are presented.
This symmetry is not enforced a-priori. How can we deal with this?

### Option 1 - help the algorithm to find symmetry statistically

Mirror all input data and add cues in pairs corresponding to a cue and its
logical complement.
Hope that the tree construction algorithnm learns the required symmetry.

### Option 2 - try to enforce the symmetry structurally

Start by defining 1-sided features:

$$
f(g^{(1)}, g^{(2)}, \mathcal{C}) \to \mathbb{R}
$$

where

* $g^{(1)} = (a^{(1)}_1, a^{(1)}_2)$
* $g^{(2)} = (a^{(2)}_1, a^{(2)}_2)$
* $\mathcal{C}$ is contextual information e.g. the ranks of the fractals.

These features can be continuous (e.g. Expected growth rate of $g^{(1)}$)
or binary (e.g. "$g^{(1)}$ contains a fractal of rank 1, 2 or 3".)
Define these 1-sided features so that they are

* non-decreasing as option $g^{(1)}$ gets better
* non-increasing as option $g^{(2)}$ gets worse.

For each 1-sided feature, we construct an associated signed cue:

$$
F(g^{(1)}, g^{(2)}, \mathcal{C}) = f(g^{(1)}, g^{(2)}, \mathcal{C}) -  f(g^{(2)}, g^{(1)}, \mathcal{C})
$$

Presence or absence of a cue in the fast-and-frugal tree is represented by an exit criterion of the form

$$
| F(g^{(1)}, g^{(2)}, \mathcal{C}) | > \tau
$$

where $\tau \geq 0$ is a threshold to be learned as part of the tree construction.

An exit then invokes the decision rule which checks the sign of the cue:
```
if $F(g^{(1)}, g^{(2)}, \mathcal{C}) > tau$
   $g^{(1)}$ is preferred
else
   $g^{(2)}$ is preferred
```
