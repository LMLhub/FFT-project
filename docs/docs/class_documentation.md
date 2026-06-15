# Class Documentation: Cue, FFT, and Experiment

## Overview

This project models gamble choices with three main classes:

* `Cue`: evaluates one decision cue for a pair of gambles.
* `FFT`: combines cues into a fast-and-frugal tree and uses them sequentially to make a choice between gamble pairs.
* `Experiment`: runs one or more FFTs over gamble data, simulates outcomes, tracks wealth, and calculates performance measures.

A gamble pair is represented by four values:

* `x_left_up`
* `x_left_down`
* `x_right_up`
* `x_right_down`

The first two values describe the left gamble, and the second two values describe the right gamble.
**Important**: Gamble pair values are gamma values, i.e. growth rates. For additive gambles this implies $x_{t+1} = x_{t} + \txt{gamma}$ and for multiplicative $x_{t+1} = x_{t} e^\txt{gamma}$

All classes have global dictionaries so that the objects can be called from anywhere in the code.

---

## Cue Class

The `Cue` class represents a single feature-based rule for comparing the left and right gambles.

Each cue wraps a feature function. The feature function is evaluated twice: once with the left gamble first and once with the right gamble first. The difference between these two evaluations determines both the cue strength and the side favored by the cue.

### Mathematical Definition

For a feature function `f`, define:

$$
F = f(G_L, G_R) - f(G_R, G_L)
$$

The cue value is:

$$
\text{cue value} = |F|
$$

The side favored by the cue is:

* `left` if the cue value is greater than the threshold and `F` is positive.
* `right` if the cue value is greater than the threshold and `F` is negative.
* `None` if the cue is not present, i.e. the value does not exceed the threshold.

Boolean cues always use a threshold of `0`. Numerical cues must be given an explicit numeric threshold.

### Constructor

```python
Cue(
    id: str,
    name: str,
    description: str,
    feature,
    type: str,
    threshold=None,
    params=None,
    required_args=list,
)
```

### Attributes

* `id`: unique cue identifier.
* `name`: short cue name.
* `description`: explanation of the cue.
* `feature`: callable feature function.
* `type`: either `"boolean"` or `"numerical"`.
* `threshold`: decision threshold. Boolean cues automatically use `0`.
* `params`: dictionary of fixed parameters passed to the feature function. Example: The set of possible fractal values can be passed to calculate absolute ranking of fractals.
* `required_args`: ordered list of required input columns.

The first four entries in `required_args` must correspond to the four gamble values. Any later entries are treated as extra arguments, such as `wealth`, and are passed to the feature function by keyword.

### Validation

When a cue is created, the class checks that:

* `feature` is callable.
* `params` is a dictionary.
* The feature function signature matches `required_args` plus `params`.
* All fixed feature parameters are present in `params`.
* Numerical cues have a numeric threshold.
* The cue `id` has not already been registered.

Created cues are stored in the class-level registry:

```python
Cue.cue_registry
```

### `evaluate`

```python
cue_value, side_if_true = cue.evaluate(
    x_left_up,
    x_left_down,
    x_right_up,
    x_right_down,
    **extra_args,
)
```

Evaluates the cue for a single gamble pair.

Returns:

* `cue_value`: absolute difference between the feature values.
* `side_if_true`: `"left"`, `"right"`, or `None`.

### `evaluate_df`

```python
result_df = cue.evaluate_df(gamble_data)
```

**NOT USED**. Evaluates the cue row by row for a pandas DataFrame. The DataFrame must contain all columns listed in `required_args`.

Adds two columns:

* `<cue_id>_value`
* `<cue_id>_side_if_true`

### `to_dict`

Returns a dictionary representation of the cue, including its id, name, description, type, threshold, parameters, feature name, and required arguments.

### `save_registry`

```python
Cue.save_registry(filepath)
```

Saves all registered cues to a YAML file as the specified filepath.

---

## FFT Class

The `FFT` class represents a fast-and-frugal tree: an ordered list of cues used to make a decision between the left and right gamble.

Cues are evaluated one at a time. The first cue that favors a side determines the final decision. If no cue favors either side, the FFT makes a random left/right choice.

### Constructor

```python
FFT(
    id,
    name,
    description,
    cues,
)
```

### Attributes

* `id`: unique FFT identifier.
* `name`: short FFT name.
* `description`: explanation of the decision rule.
* `cues`: ordered list of `Cue` objects.
* `tree_length`: number of cues in the tree.

Created FFTs are stored in the class-level registry:

```python
FFT.FFT_registry
```

### Decision Process

For cues \(C_1, C_2, \dots, C_n\):

1. Evaluate cue \(C_i\).
2. If the cue returns `left` or `right`, stop and return that side.
3. If the cue returns `None`, continue to the next cue.
4. If no cue decides, choose randomly between `left` and `right`.

### `decide`

```python
cue_values, side, cues_used = fft.decide(
    x_left_up,
    x_left_down,
    x_right_up,
    x_right_down,
    **kwargs,
)
```

Evaluates the FFT for a single gamble pair.

Returns:

* `cue_values`: list of cue values evaluated before stopping.
* `side`: final decision, either `"left"` or `"right"`.
* `cues_used`: number of cues used to reach the decision.

If no cue makes a decision, `cues_used` is `len(cues) + 1` because the final random choice is counted as an additional decision step.

### `decide_df`

```python
result_df = fft.decide_df(gamble_data, required_args)
```
**NOT USED**. Evaluates the FFT row by row for a DataFrame.

`required_args` must contain at least the four gamble-value column names, followed by any extra columns required by the cues.

Adds columns:

* `<cue_id>_cue_value` for each evaluated cue.
* `<fft_id>_decision`
* `<fft_id>_cues_used`

### `retrieve_required_args`

```python
required_args = fft.retrieve_required_args()
```

Returns the union of all `required_args` needed by the cues in the FFT, preserving the order in which arguments are first encountered.

### `to_dict`

Returns a dictionary representation of the FFT, including its id, name, description, cue ids, cue names, and tree length.

### `save_registry`

```python
FFT.save_registry(filepath)
```

Saves all registered FFTs to a YAML file at the specified path.

---

## Experiment Class

The `Experiment` class runs one or more FFTs over a sequence of gamble pairs. It simulates choices, random outcomes, and wealth trajectories.

An experiment is useful when evaluating how different FFTs behave across the same gamble data.

### Constructor

```python
Experiment(
    id,
    name,
    description,
    ffts,
    dynamic=None,
    gamble_data=None,
    initial_wealth=None,
)
```

### Attributes

* `id`: unique experiment identifier.
* `name`: short experiment name.
* `description`: explanation of the experiment.
* `ffts`: list of `FFT` objects evaluated in the experiment.
* `dynamic`: wealth update rule, either `"additive"` or `"multiplicative"`.
* `gamble_data`: pandas DataFrame containing gamble values and any extra cue arguments.
* `initial_wealth`: starting wealth for each FFT run.
* `required_args`: required columns collected from all FFTs.
* `random_seeds`: list of random seeds used across runs.
* `runs`: number of completed runs.
* `results`: accumulated results from all runs.

Created experiments are stored in the class-level registry:

```python
Experiment.experiment_registry
```

### Validation

When an experiment is created, the class checks that:

* `ffts` is a list.
* Every item in `ffts` is an `FFT` instance.
* `dynamic` is either `"multiplicative"` or `"additive"`.
* `initial_wealth` is provided.
* `gamble_data` is provided.
* `gamble_data` contains all required columns for the FFT cues.
* The experiment `id` has not already been registered.

### `run_experiment`

```python
results = experiment.run_experiment(
    initial_wealth=None,
    random_seed=None,
)
```

Runs every FFT through every row of `gamble_data`.

For each FFT and each gamble:

1. The FFT chooses `"left"` or `"right"`.
2. The experiment randomly draws an outcome, either `"up"` or `"down"`.
3. The payoff is selected from the chosen side and realized outcome.
4. Wealth is updated using the experiment dynamic.

For additive dynamics:

$$
w_{t+1} = w_t + \text{payoff}
$$

For multiplicative dynamics:

$$
w_{t+1} = w_t \cdot e^{\text{payoff}}
$$

If `random_seed` is not provided, the experiment generates one automatically. If `initial_wealth` is not provided to the method, it uses the experiment's default `initial_wealth`.

Returns `self.results`, a DataFrame that accumulates all runs.

### Results Format

The returned results DataFrame uses the original gamble index as rows. The columns are a three-level MultiIndex:

```text
(fft_id, run, metric)
```

The stored metrics are:

* `decision`: side chosen by the FFT.
* `cues_used`: number of cues used by the FFT.
* `outcome`: realized random outcome, `"up"` or `"down"`.
* `wealth`: wealth after the gamble is resolved.

Running the experiment multiple times appends additional run columns to `self.results`.

### `accuracy`

```python
accuracy = experiment.accuracy(
    fft_id,
    reference_id,
    run_no=None,
)
```

Calculates the proportion of decisions made by `fft_id` that match the decisions made by `reference_id`.

If `run_no` is:

* `None`: evaluate all runs.
* an integer: evaluate that run only.
* a list or iterable: evaluate the specified runs.

Returns a float between `0` and `1`.

### `frugality`

```python
frugality = experiment.frugality(
    fft_id,
    run_no=None,
)
```

Calculates the average number of cues used by an FFT.

If `run_no` is:

* `None`: evaluate all runs.
* an integer: evaluate that run only.
* a list or iterable: evaluate the specified runs.

Lower values indicate more frugal decision-making.

---

## Typical Workflow

1. Define feature functions.
2. Create `Cue` objects from those feature functions.
3. Build one or more `FFT` objects from ordered cue lists.
4. Prepare a `gamble_data` DataFrame with the required columns.
5. Create an `Experiment`.
6. Run the experiment and analyze `accuracy` and `frugality`.

Example:

```python
from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import growth_rate

cue = Cue(
    id="growth",
    name="Growth Rate",
    description="Chooses the gamble with the higher growth rate.",
    feature=growth_rate,
    type="numerical",
    threshold=0,
    required_args=[
        "gamma_left_up",
        "gamma_left_down",
        "gamma_right_up",
        "gamma_right_down",
    ],
)

fft = FFT(
    id="growth_fft",
    name="Growth-rate FFT",
    description="A one-cue FFT based on growth rate.",
    cues=[cue],
)

experiment = Experiment(
    id="exp_growth",
    name="Growth-rate experiment",
    description="Runs a growth-rate FFT over gamble data.",
    ffts=[fft],
    dynamic="additive",
    gamble_data=gamble_data,
    initial_wealth=1000,
)

results = experiment.run_experiment(random_seed=42)
```
