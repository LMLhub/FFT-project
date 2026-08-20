# Accuracy measures

The `Experiment.accuracy` method compares the choices made by one FFT with
those made by a reference FFT over the selected run or runs.

## Accuracy

Ordinary accuracy gives every gamble pair equal importance. If $m_i$ is 1
when the two FFTs make the same choice on trial $i$, and 0 otherwise, then

$$
A = \frac{\sum_i m_i}{N}.
$$

An accuracy of 1 means that every choice agrees, 0 means that every choice
disagrees, and 0.5 is the expected value for random binary choices.

```python
experiment.accuracy("fft", "reference_fft")
```

## Importance-weighted accuracy

Importance-weighted accuracy gives more influence to gamble pairs whose two
options have substantially different expected gamma values. For a gamble with
equally likely up and down outcomes, the expected gammas are

$$
\bar{\gamma}_{L,i}
= \frac{\gamma_{L,i}^{up} + \gamma_{L,i}^{down}}{2},
\qquad
\bar{\gamma}_{R,i}
= \frac{\gamma_{R,i}^{up} + \gamma_{R,i}^{down}}{2}.
$$

The trial's normalized importance weight is

$$
w_i =
\frac{|\bar{\gamma}_{L,i} - \bar{\gamma}_{R,i}|}
{\max_j |\bar{\gamma}_{L,j} - \bar{\gamma}_{R,j}|}.
$$

Let $s_i=1$ for a matching choice and $s_i=-1$ for a mismatch. The weighted
accuracy is

$$
A_w = \frac{1}{2}\left(1 +
\frac{\sum_i s_i w_i}{\sum_i w_i}\right).
$$

Normalizing by the total weight keeps this measure on the same scale as
ordinary accuracy: 0 represents complete disagreement, 0.5 is the random
baseline, and 1 represents complete agreement. The weighted result can be
higher or lower than ordinary accuracy depending on whether agreement is
better on high-importance trials. If every gamble has zero importance weight,
the method returns 0.5.

```python
experiment.accuracy(
    "fft",
    "reference_fft",
    importance_weighted=True,
)
```
