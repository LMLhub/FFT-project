# Sequential Preference Learning Using Pareto-Optimal Decision Rules

## 1. Problem statement

We wish to construct short, interpretable sequences of cue-based decision rules for a binary preference learning task.

We have:

- $N$ instances $(A, B)$, each with a known "correct" choice between the two options, $A$ or $B$.
- $M$ fixed cue-based decision rules, each of which can output $A$, $B$, or abstain for any choice instance $(A, B)$. The implementation of cue-based decision rules ensures symmetry with respect to $A \leftrightarrow B$.
- A maximum sequence length $K$, initially expected to be 3 or 4.

Our objective is to identify sequences of rules that achieve a desirable balance between decision accuracy and coverage.

Individual rules may be highly accurate but choose for relatively few instances, while others may choose for many instances less accurately. Rather than selecting rules greedily according to a single performance metric, we propose a **heuristic search based on local Pareto optimality in the accuracy–coverage plane**.



## 2. Terminology and mathematical formulation

### Dataset

Represent the data by a prediction matrix

$$
X\in \left\{-1,0,1\right\}^{N\times M},
$$

with encoding

$$
X_{ij}=\begin{cases}
1 & \text{rule }j\text{ chooses A},\\
0 & \text{rule }j\text{ abstains},\\
-1 & \text{rule }j\text{ chooses B}.
\end{cases}
$$

The correct labels are stored in

$$
y\in\left\{1,-1\right\}^{N}.
$$

The decision rules are fixed: the algorithm does not learn or modify them. It searches for effective combinations and orderings.

### Decision sequence

A **decision sequence** is an ordered tuple of rule indices,

$$
\sigma=(j_1,j_2,\ldots,j_k),\qquad k\leq K.
$$

The rules are applied sequentially. The **first rule that does not abstain** determines the final choice for an instance. If every rule abstains, the instance remains undecided.

Unlike a conventional FFT, only the abstention branch continues. The structure can also be called a **decision list**, **rule cascade**, or **sequential decision tree with abstention**.

### Residual dataset

At stage $k$, the **residual dataset** $S_k$ contains the instances not decided by any preceding rule. Initially,

$$
S_1=\{1,\ldots,N\}.
$$

After applying rule $j_k$,

$$
S_{k+1}=\{i\in S_k:X_{i,j_k}=0\}.
$$

Each residual dataset will be represented by a Boolean mask of length $N$.

### Decision rule (cue) performance

For a candidate cue-based decision rule $j$ evaluated on residual dataset $S$, define:

| Metric | Definition |
|---|---|
| Correct $C_j(S)$ | Number of correct choices |
| Incorrect $E_j(S)$ | Number of incorrect choices |
| Decided $D_j(S)$ | $C_j(S)+E_j(S)$ |
| Coverage $c_j(S)$ | $D_j(S)/\lvert S\rvert$ |
| Accuracy $a_j(S)$ | $C_j(S)/D_j(S)$ |

Accuracy is undefined if $D_j(S)=0$; such rules will be excluded from candidate selection. If $S$ is empty, the search state is terminal.

Coverage and accuracy are measured relative to the **current residual dataset**, not necessarily the original dataset.

### Local Pareto frontier

Rule $j$ is **locally Pareto optimal** on $S$ if no other available rule $\ell$ satisfies both

$$
a_\ell(S)\geq a_j(S),\qquad c_\ell(S)\geq c_j(S),
$$

with at least one strict inequality.

The set of all such rules is the **local Pareto frontier**. It is recomputed for each search state because the residual datasets differ across branches.

Pareto selection is a **heuristic search-space reduction**. Local dominance does not imply global dominance: rules with apparently inferior immediate accuracy and coverage may leave residual instances that subsequent rules decide upon more effectively.

## 3. Proposed search algorithm

We propose a **breadth-first search** that recursively expands candidate sequences using the local Pareto frontier.

```text
Initial state (empty sequence; all instances remain)
    |
    v
Evaluate available rules on the residual dataset
    |
    v
Identify locally Pareto-optimal rules
    |
    v
Create one child search state per retained rule
    |
    +---- Repeat for each branch until depth K or termination
    |
    v
Evaluate and rank all retained candidate sequences
```

Procedure:

1. Initialise an empty sequence with a residual mask containing all instances.
2. Evaluate available rules on the current residual dataset.
3. Identify the local Pareto frontier.
4. Create a new search state for each rule on that frontier.
5. Update its residual mask and cumulative correct/incorrect counts.
6. Repeat until maximum depth $K$, no instances remain, or no further useful choices are possible.
7. Evaluate and rank the generated sequences.

Candidate sequences are retained at **every depth**, not only depth $K$. Initially, each rule may appear at most once in a sequence.

## 4. Search state and data structures


### Dataset

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    predictions: np.ndarray  # shape (N, M); values -1, 0, 1
    labels: np.ndarray       # shape (N,); values 0, 1
    feature_names: list[str]
```

The dataset is created before anything else using the existing code infrastructure
and is shared by all search states and remains unchanged during the search.

### Search state

```python
@dataclass
class SearchState:
    sequence: tuple[int, ...]
    remaining: np.ndarray    # Boolean mask, length N
    correct: int = 0
    incorrect: int = 0
```

The state records the selected sequence, the residual dataset, and cumulative performance. The number of undecided instances is `remaining.sum()`.

Search states will be treated as **immutable snapshots**: extending a sequence creates a new mask rather than modifying an existing mask in place.

### State transition

The core operation, `extend_state(state, feature, dataset)`, will:

1. Identify remaining instances undecided by the chosen rule.
2. Count its correct and incorrect choices.
3. Add those counts to the cumulative totals.
4. Create a new Boolean residual mask.
5. Return a new search state.

For example:

```python
def extend_state(state: SearchState, feature: int, dataset: Dataset) -> SearchState:
    predictions = dataset.predictions[:, feature]
    decided = state.remaining & (predictions != 0)
    correct = np.count_nonzero(decided & (predictions == dataset.labels))
    incorrect = np.count_nonzero(decided) - correct

    return SearchState(
        sequence=state.sequence + (feature,),
        remaining=state.remaining & (predictions == 0),
        correct=state.correct + int(correct),
        incorrect=state.incorrect + int(incorrect),
    )
```

The transition function does not need to know anything about Pareto selection or global scoring.

## 5. Global sequence evaluation

For a sequence $\sigma$, let $C(\sigma)$, $E(\sigma)$, and $U(\sigma)$ be the numbers of correct, incorrect, and undecided instances. Then

$$
C(\sigma)+E(\sigma)+U(\sigma)=N.
$$

One candidate scalar objective is

$$
\boxed{J(\sigma)=C(\sigma)-\lambda E(\sigma)-\mu U(\sigma)},
$$

where $\lambda$ and $\mu$ specify penalties for errors and abstentions. The exact objective is not yet fixed.

We will store raw decision counts independently of the scoring function so that alternative objectives can be compared **without rerunning the search**.

### Optional terminal rule

We may append a special fallback rule that never abstains. An unbiased random binary assignment would decide on all remaining instances with expected accuracy $1/2$. Thus,

$$
\mathbb{E}[C_{\mathrm{final}}]=C(\sigma)+\frac{U(\sigma)}{2},
$$

$$
\mathbb{E}[E_{\mathrm{final}}]=E(\sigma)+\frac{U(\sigma)}{2}.
$$

The fallback will be handled separately from the ordinary sequence, so its contribution remains identifiable. Expected fallback performance can be computed analytically, without Monte Carlo noise.

## 6. Code architecture

The implementation will be divided into five modules which will be
located in ```src/fft_project/sequential_decision_rules```:

| Module | Responsibility |
|---|---|
| `data.py` | Dataset representation, validation, and loading |
| `metrics.py` | Rule accuracy and coverage on residual datasets |
| `pareto.py` | Identification of nondominated rules |
| `search.py` | Search states, transitions, and sequence generation |
| `evaluation.py` | Global scoring and comparison of sequences |

This should allow the code to be separated out into a more general package later
if it turns out out to be useful.

Provisional interfaces:

```python
evaluate_features(dataset, remaining, available_features)
pareto_frontier(feature_metrics)
extend_state(state, feature, dataset)
search(dataset, max_depth)
score(sequence_result, parameters)
```

