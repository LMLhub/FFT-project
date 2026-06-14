# Eta Estimation Method

To use the isoelastic utility function as a cue, a value for the risk
aversion parameter $\eta$ is needed. Rather than fixing $\eta$ manually,
it can be estimated from observed human choices between gamble pairs. The
core idea is to find the $\eta$ that best explains why a participant chose
the way they did across many trials.

## The utility function

The isoelastic (CRRA) family, parameterised by $\eta$, is given by:

$$
f_\eta(x) = \begin{cases} \dfrac{x^{1-\eta} - 1}{1 - \eta} & \text{for } \eta \neq 1 \\ \ln x & \text{for } \eta = 1 \end{cases}
$$

The parameter $\eta$ controls the curvature of this transformation and
therefore how a participant values outcomes. At $\eta = 0$ the
transformation is linear — the participant cares about the expected
value of a gamble and treats gains and losses symmetrically. At $\eta = 1$
it becomes logarithmic, which is the growth-optimal strategy under
multiplicative dynamics. As $\eta$ increases beyond 1, large outcomes are
progressively underweighted, reflecting stronger risk aversion. Negative
values of $\eta$ produce the opposite effect, corresponding to
risk-seeking behaviour.

## Scope: which cue does eta estimation apply to?

Eta belongs to one cue, the expected isoelastic utility cue. So what we estimate
here is the eta for that cue alone, not for a whole tree.

The value that decides the choice is the difference in expected transformed
wealth between the two gambles:

$$
F_t(\eta) = f_\eta(\gamma_L)_t - f_\eta(\gamma_R)_t
$$

This is the same thing written as $\Delta\langle\delta f_\eta\rangle$ further
down.

A fast-and-frugal tree decides in like this, once a cue fires, it picks the side with the higher value. What we use here is a softer version of that, the participant picks the better side most of the time, but not always. How strict this is depends on beta, the bigger beta gets, the closer we are to the hard rule of the tree. In that sense the tree is just the extreme case.

This only holds  as long as there is a single cue. Estimating eta like this
assumes the utility cue is the only thing behind a choice. As soon as a tree
uses several cues together (say the utility cue plus something like avoid worst),
we'd need to decide separately how to handle that, and that goes beyond what eta
estimation covers here.

## Modelling choices probabilistically

The probability of choosing the left gamble is modelled as a logistic function
of the difference in expected transformed wealth between the two options:

$$
\theta\bigl(\Delta\langle\delta f_\eta\rangle\bigr) = \frac{1}{1 + e^{-\beta\,\Delta\langle\delta f_\eta\rangle}}
$$

where $\Delta\langle\delta f_\eta\rangle = \langle\delta f_\eta(\gamma_L)\rangle - \langle\delta f_\eta(\gamma_R)\rangle$
is the difference in expected transformed wealth between the left and
right gamble for a given $\eta$. The sensitivity parameter $\beta$
controls how reliably the participant acts on this difference: a large
$\beta$ means the participant almost always picks the better option, while
a small $\beta$ means choices are close to random. Both $\eta$ and $\beta$
are unknown and estimated simultaneously.

## From single choices to the likelihood

The logistic function gives the probability $\theta_t$ of a single choice. To
estimate $\eta$, the probability of the entire sequence of observed choices is
needed — and the Bernoulli distribution provides the link.

Each trial has exactly two outcomes (left or right), so each choice is modelled
as a Bernoulli trial:

$$
y_t \sim \text{Bernoulli}(\theta_t)
$$

where $y_t = 1$ if the participant chose left and $y_t = 0$ if they chose right.
The probability of a single observed choice is:

$$
p(y_t \mid \eta, \beta) = \theta_t^{\,y_t}\,(1-\theta_t)^{\,1-y_t}
$$

The exponents act as a switch: if $y_t = 1$ the term reduces to $\theta_t$, and
if $y_t = 0$ it reduces to $1-\theta_t$. Assuming choices are conditionally
independent given the parameters, the likelihood of all $T$ choices is the
product:

$$
p(\text{data} \mid \eta, \beta) = \prod_{t=1}^{T} \theta_t^{\,y_t}\,(1-\theta_t)^{\,1-y_t}
$$

In practice the **log-likelihood** is used to avoid numerical underflow, turning
the product into a sum:

$$
\log p(\text{data} \mid \eta, \beta) = \sum_{t=1}^{T}\big[\,y_t\log\theta_t + (1-y_t)\log(1-\theta_t)\,\big]
$$

This is the term $p(\text{data} \mid \eta, \beta)$ that enters Bayes' theorem
below.

## Bayesian estimation

The goal is to find which values of $\eta$ are consistent with the
observed choices. Bayesian inference does this by combining prior knowledge
and the data into a posterior distribution that reflects all plausible values
of $\eta$ along with their relative probability.

$$
p(\eta, \beta \mid \text{data}) \propto p(\text{data} \mid \eta, \beta) \times p(\eta, \beta)
$$

The **prior** $p(\eta, \beta)$ represents what is assumed before seeing
any choices. Weakly informative priors are used — $\eta$ is normally
distributed over a broad range, $\beta$ is log-normally distributed —
so that the estimates are driven by the data rather than prior assumptions.

The **likelihood** $p(\text{data} \mid \eta, \beta)$ is the Bernoulli product
defined above. It measures how well a candidate $(\eta, \beta)$ pair predicts
the participant's actual choices. If a particular $\eta$ consistently predicts
the gamble the participant chose, it receives a high likelihood.

The **posterior** is the combination of both. Rather than returning a
single number, it is a full distribution showing which $\eta$ values are
plausible given the data. The peak of this distribution — the Maximum a
Posteriori (MAP) estimate — is used as the point estimate per participant.

## MCMC sampling

The posterior has no closed-form solution and is instead approximated
using Markov Chain Monte Carlo (MCMC) sampling via JAGS. The sampler
generates a large collection of $(\eta, \beta)$ values by exploring the
parameter space, spending more time in regions where the parameters fit
the data well. Four independent chains with 10,000 samples each are run,
discarding the first 1,000 as burn-in. Convergence is verified using the
Gelman-Rubin R-hat statistic, which should fall between 1 and 1.01.

## Hierarchical estimation

Rather than estimating $\eta$ for each participant completely
independently, a hierarchical (partial pooling) model is used. All
individual $\eta$ values are assumed to come from the same group-level
distribution, whose mean and variance are estimated from the full dataset.
As a result, each participant's estimate is informed not only by their own
choices, but also by what is typical across the group. This is
particularly useful when a participant's choices are noisy: instead of
producing an unreliable individual estimate, the model falls back on the
group mean. This process is known as shrinkage.
