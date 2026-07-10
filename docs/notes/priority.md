## Notes about priority heuristic

For the priority heuristic, we consider the choices between gambles that are represented in terms of changes in wealth (also called gains or losses):

$$
G^{(A)} = \left\{
\begin{array}{ll}
(\Delta x)^{(A)}_1 & \text{with $p=\frac{1}{2}$}\\
(\Delta x)^{(A)}_2 & \text{with $p=\frac{1}{2}$}
\end{array}
\right.
$$

$$
G^{(B)} = \left\{
\begin{array}{ll}
(\Delta x)^{(B)}_1 & \text{with $p=\frac{1}{2}$}\\
(\Delta x)^{(B)}_2 & \text{with $p=\frac{1}{2}$}
\end{array}
\right.
$$

There are potentially different variants of the priority heuristic.
The original version of priority heuristic is described by Brandstätter, Gigerenzer, and Hertwig (2006). They seperate the decision rule for gambles with gains and gambles with losses. They write:

> Priority Rule. Go through reasons in the order: minimum
gain, probability of minimum gain, maximum gain. 

> Stopping Rule. Stop examination if the minimum gains differ
by 1/10 (or more) of the maximum gain; otherwise, stop
examination if probabilities differ by 1/10 (or more) of the
probability scale.

> Decision Rule. Choose the gamble with the more attractive
gain (probability). The term attractive refers to the gamble with the higher (minimum
or maximum) gain and the lower probability of the minimum gain.

For losses the heuristic is formulated:

> Priority Rule. Go through reasons in the order: minimum loss,
probability of minimum loss, maximum loss.

> Stopping Rule. Stop examination if the minimum losses differ
by 1/10 (or more) of the maximum loss; otherwise, stop
examination if probabilities differ by 1/10 (or more) of the
probability scale.

> Decision Rule. Choose the gamble with the more attractive
loss (probability). The term attractive refers to the gamble with the lower (minimum
or maximum) loss and the higher probability of the minimum loss.

### Version 1 (closest to the original)
In our implementation we replace the 1/10 in the rule with a tolerance $\tau$ that can be set when creating the cue.
We define $m^i$ as the minimum gain/loss and $M^i$ as the maximum gain/loss of gamble option $i \in \{A,B\}$:

$$ m^{i} = \min_{} ((\Delta x)^{i}_1, (\Delta x)^{i}_2) $$

$$ M^{i} = \max_{} ((\Delta x)^{i}_1, (\Delta x)^{i}_2) $$ 

The rule implies three cues to be evaluated in order.

#### Pre-processing
For the multiplicative case, all fractal values must be turned into changes in wealth. Since wealth cancels out in all equations in this decision rule, it is sufficient to represent changes in wealth by their multiplicative factor, i.e. we have $\Delta x = e^g -1$ where $g$ is the "gamma" value of the fractal.

In the additive case, fractal values are represented by changes in wealth and hence $\Delta x = g$, where $g$ is the fractal "gamma" value.

Since the choices in the experiment rarely concerns gambles with purely gains or losses, we need a method to decide when to use the rule for gains and when to use the rule for losses.

In version 1, we use the gains rule whenever the average outcome of all four fractals is positive, and the loss rule if negative:

If $\left( (\Delta x)^{(A)}_1 + (\Delta x)^{(A)}_2 + (\Delta x)^{(B)}_1 + (\Delta x)^{(B)}_2 \right) / 4 > 0$
then use 'gains' rule, otherwise use 'losses' 

#### Cue 1
Cue 1 is present if the difference in minimum gains (losses) are greater than the specified tolerance scaled by the maximum of outcome the two gambles.

If $\left| m^{(A)} - m^{(B)} \right| > \tau \max_{} (M^{(A)},M^{(B)})$ then:

- If gains: If $m^{A} > m^{B}$, choose gamble $A$ otherwise $B$ (i.e. pick the gamble with the highest minimum gain).

- If losses: If $m^{A} < m^{B}$, choose gamble $A$ otherwise $B$ (i.e. pick the gamble with the lowest minimum losses).

#### Cue 2
This cue looks at differences in probability - since all probabilities are the same, we skip this cue.

#### Cue 3
This is the final cue that always lead to a decision (it is therefore always present):

- If gains: If $M^{A} > M^{B}$, choose gamble $A$ otherwise $B$ (i.e. choose the gamble with the highest maximum gains)

- If losses: If $M^{A} < M^{B}$, choose gamble $A$ otherwise $B$ (i.e. choose the gamble with the lowest maximum losses).

### Version 2 (no loss)
The no-loss version of the priority heurstic does not distinguish between gains- and loss-type gambles pairs. All gambles are treated as gains.

#### Pre-processing
To make sure there are no losses in the gambles, the preprocessing indentifies the smallest value, $m$ and subtraact this value from all values.

$$
m = \min_{} ((\Delta x)^{A}_1, (\Delta x)^{A}_2, (\Delta x)^{B}_1, (\Delta x)^{B}_2 )
$$

We get the modified gambles:

$$
\tilde{G}^{(A)} = \left\{
\begin{array}{ll}
(\Delta x)^{(A)}_1 - m & \text{with $p=\frac{1}{2}$}\\
(\Delta x)^{(A)}_2 - m & \text{with $p=\frac{1}{2}$}
\end{array}
\right.
$$

$$
\tilde{G}^{(B)} = \left\{
\begin{array}{ll}
(\Delta x)^{(B)}_1 - m & \text{with $p=\frac{1}{2}$}\\
(\Delta x)^{(B)}_2 - m & \text{with $p=\frac{1}{2}$}
\end{array}
\right.
$$

For the modified gambles, all wealth outcomes are positive and the minimum outcome is always 0.
After this preprocessing, the heuristic takes the modified gambles and proceeds with the steps described in version 1.

The idea behind this version is (1) to avoid the 'loss aversion' implied by the assymetric rule for gains and losses, and (2) use the distance between the maximum and minimum (instead of the maximum alone) when evaluating the first cue.

### Version 3 (ranks)
This version uses the same decision rule as version 1, but gambles are defined by their fractal rank values, $(r)^{i}_j$, instead of the changes in wealth, $(\Delta x)^{i}_j$. Higher rank is better.