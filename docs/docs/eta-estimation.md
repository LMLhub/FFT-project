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
transformation is linear, so the participant cares about the expected
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

A fast-and-frugal tree decides like this: once a cue fires, it picks the side with the higher value. What we use here is a softer version of that, the participant picks the better side most of the time, but not always. How strict this is depends on beta, the bigger beta gets, the closer we are to the hard rule of the tree. In that sense the tree is just the extreme case.

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
needed, and the Bernoulli distribution provides the link.

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

## Maximum likelihood estimate

We estimate $\eta$ and $\beta$ by maximising the log-likelihood above:

$$
(\hat\eta, \hat\beta) = \arg\max_{\eta,\,\beta}\; \log p(\text{data} \mid \eta, \beta)
$$

There is no closed form, but with only two parameters a numerical optimiser
(Nelder-Mead) converges quickly. We optimise over $\log\beta$ to keep $\beta$
positive and fit each participant separately, giving one $\hat\eta$ per
participant and condition. $\beta$ acts as a nuisance parameter that absorbs
choice noise, so $\hat\eta$ reflects the preferred side rather than the
consistency of the choices.

## MAP with weak priors

For near-separable choices the likelihood is maximised as $\beta \to \infty$
with $\eta$ diverging. We regularise by maximising the log-posterior instead,

$$
(\hat\eta, \hat\beta) = \arg\max_{\eta,\,\beta}\;\big[\log p(\text{data} \mid \eta, \beta) + \log p(\eta, \beta)\big]
$$

with weak priors (a broad normal on $\eta$, a log-normal on $\beta$).
Dropping the priors recovers the MLE.

## What we leave out for now

We use the same likelihood as the paper but skip the hierarchical Bayesian
model and MCMC (JAGS). That would add full posteriors per participant and
partial pooling across participants (shrinkage for noisy participants).
Neither is needed for a first per-participant point estimate, so we defer it.
The hierarchical model can be added later on top of the same likelihood.
