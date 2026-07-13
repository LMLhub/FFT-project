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

## Why and when does it work?
This section includes some notes on the relationship between the priority heuristic and growth rate optimality.

We generalise the gamble to allow for different probabilities:

$$
G^{(A)} = \left\{
\begin{array}{ll}
m^{A} & \text{with $p$}\\
M^{A} & \text{with $1-p$}
\end{array}
\right.
$$

$$
G^{(B)} = \left\{
\begin{array}{ll}
m^{B} & \text{with $q$}\\
M^{B} & \text{with $1-q$}
\end{array}
\right.
$$

where $m$ and $M$ refer to the minimum and maximum change in wealth, respectively. For simplicity, let's further assume that $A$ is the less risky choice, i.e. $m^B < m^A < M^A < M^B$. 

### Additive dynamics
#### Cue 1
Under additive dynamic the growth rate optimal choice can be formulated as follows: If

$$
p m^A + (1-p)M^A \geq q m^B + (1-q)M^B
$$

then choose $A$.

This condition is equivalent to 

$$
(p m^A - q m^B) \geq  ((1-q)M^B - (1-p)M^A)
$$

$$
m^A - \frac{q}{p} m^B \geq \frac{(1-q)}{p} M^B  - \frac{(1-p)}{p} M^A  \qquad \qquad (1)
$$

If $p \approx q$, we can get an expression that is equivalent to Cue 1 of the priority heuristic:

$$
m^A - m^B \geq \frac{(1-p)}{p} (M^B - M^A)  \qquad \qquad (2) 
$$

$$
m^A - m^B \geq  \frac{(1-p)}{p} (1-t)M^B
$$

where in the last inequality, we are expressing $M^A$ as a fraction of $M^B$ using $t$: $M^A = tM^B$. This inequality is identical to the priority heuristic with $\frac{(1-p)}{p}(1-t) = \tau$ as we know that $M^B$ is the maximum gains in the gamble pairs.

##### Example
Example: In the choice between 100 for sure and 1000 with 10 percent, 0 with 90 percent chance, we can calculate the tolerance to be $(1-0.9)/(0.9)*(1-0.1) = 0.1$ which is the tolerance level suggested in the original heuristic.

##### ErgEx
In the experiment, $p = q = \frac{1}{2}$, which means that the tolerance level depends only on the relationship between $M^A$ and $M^B$, i.e. $\tau = 1-t$ where $M^A = tM^B$.

With the experimental design conditions we know that fractal values are linearily spaced, that $m_B < m_A < M^A < M^B$ and that the fractal values are centered around 0, it is reasonable to assume that $M^A \approx 0.5 M^B$ meaning $\tau \approx 0.5$

#### Cue 2 and 3
Cue 1 may not perform well if $p$ and $q$ are very different.
Everything else equal, variations in $p$ and $q$ may lead to 

If $q<p$ then $m^A-m^B < m^A - \frac{q}{p} m^B$. This may imply that we (correcly) reject selecting $A$ (cue value is low) and move on to the next cue. Cue 2, which tells us to look at the difference in probability of the minimum gain ($p-q$). If $p-q >$ tolerance, then choose the gamble with the lowest probability of minimum gain, in this case that is $B$, because $q<p$. Even if the difference is not large enough for Cue 2 to be present, Cue 3 will ensure the selection of B (highest maximum gain).

If $q>p$ then $m^A-m^B > m^A - \frac{q}{p} m^B$ and there is a risk of choosing A when B may be better.
However, in this case, it is worth thinking of what happens with the right hand side of the ineqality condition that corresponds to cue 2. The tolerance is proportional to $\frac{1-q}{p}$ which declines as $p$ decreases. A 'fixed' tolerance would therefore suggest a too high threshold for cue 1, which again reduces the risk of a false positive. However, Cue 2 would also imply choosing A if present. Only if the difference in probability is not large enough to lead to a decision, Cue 3 will lead to the choice of B.

### Multiplicative dynamics
The relationship between the two decision rules is more complicated. However it should be clear that due to the linear spacing of time-average growth rates in the multiplicative case, the tolerance for cue 1 should be somewhat higher ($M^B = 0.5 M^A$ is no longer a good assumption, due to exponential growth of changes in wealth).
