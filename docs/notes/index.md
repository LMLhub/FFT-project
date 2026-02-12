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

