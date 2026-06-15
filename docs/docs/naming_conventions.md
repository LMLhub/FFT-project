# Naming Conventions

This document defines naming conventions for the main classes in the project:

* `Cue`
* `FFT`
* `Experiment`

The goal is to keep object IDs, display names, DataFrame columns, and generated result columns predictable across scripts, experiments, and saved registries.

---

## General Rules

Use `snake_case` for machine-readable names:

```text
growth_rate
avoid_worst_1
gamma_left_up
```

Use short, readable title case for display names:

```text
Growth Rate
Avoid Worst 1
```

Avoid spaces, capital letters, and punctuation in IDs. IDs are used in registries and generated column names, so they should be stable and easy to parse.

---

## Cue Names

### Cue IDs

Cue IDs should be descriptive, short, use abrevations and `snake_case`.

Recommended pattern:

```text
<cue_family>_<variant>
```

Examples:

```text
gr
eu_1_5
aw_1
aw_3
pb_2
fs
```
where
gr means (time average) growth rate,
eu means expected utility,
aw means avoid worst,
pb means prefer best
fs means fractal signs.

Use numeric suffixes only when the number is meaningful. For example, `avoid_worst_1` is better than `c03` because it explains the cue behavior.

If the cue works only for specific dynamics (including when fractal values are passed as additional parameters to the cue) add `_m` or `_a` to the id.

Avoid:

```text
c01
Cue1
myCue
avoid worst
```

### Cue Display Names

Cue display names should be human-readable and concise.

Examples:

```text
Time-Average Growth Rate
Expected Utility eta=1.5 - Additive
Avoid Worst 1 - Additive
Prefer Best 2 - Multiplicative
Fractal Signs - Additive
```

### Cue Descriptions

Descriptions should say what the cue compares and when it favors a side.

Example:

```text
Prefers the gamble with the higher expected isoelastic utility for eta=1.5.
```

## FFT Names

### FFT IDs

Use `fft_` and the cue ids in order without dynamics subscript. Then add the dynamics subscribt in the end.

Examples:

```text
fft_gr
fft_aw_3_pb_2_a
fft_eu_1_5_m
```

Avoid:

```text
fft1
fft2
treeA
test_fft
```

### FFT Display Names

FFT display names should describe the decision strategy in natural language.

Examples:

```text
FFT Growth-Rate
FFT Avoid-Worst 3 then Prefer Best 2 - Additive
FFT Expected Utility - Multiplicative
```

---

## Experiment Names

### Experiment Class Name

The Python class should be named:

```python
Experiment
```

Class names use `PascalCase`.

### Experiment IDs

Experiment IDs do not need to be meaningful as there are simply too many settings.

Recommended pattern:

```text
exp_<number>
```

Examples:

```text
exp_01
exp_02
```


### Experiment Display Names

Experiment display names should be readable and specific.

Examples:

```text
Experiment with Simulated Data - Additive
```

## Gamble Data Columns

Recommended gamble-value columns:

```text
gamma_left_up
gamma_left_down
gamma_right_up
gamma_right_down
```

No other fractal value inputs are allowed.

### Required Argument Order

For cues and FFTs, the first four `required_args` must always be ordered as:

```python
[
    "gamma_left_up",
    "gamma_left_down",
    "gamma_right_up",
    "gamma_right_down",
]
```

Any additional arguments come after the four gamble columns:

```python
[
    "gamma_left_up",
    "gamma_left_down",
    "gamma_right_up",
    "gamma_right_down",
    "wealth",
]
```

---

## Feature Function Names

Feature functions should use `snake_case` and describe what they compute.

Examples:

```python
growth_rate
expected_isoelastic_utility
avoid_worst_n_ranks
fractal_signs
```

Feature function parameters should also use `snake_case`:

```python
dynamic
eta
wealth
fractal_values
```

Feature function gamble arguments should use generic names because the cue evaluates both directions:

```python
g1_up
g1_down
g2_up
g2_down
```

---