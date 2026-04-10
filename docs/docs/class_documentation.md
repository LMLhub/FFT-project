# Documentation: Cue and FFT classes

## Overview

The Cue and FFT classes implement a framework for modeling decision-making using **cues** and **fast-and-frugal trees (FFT)**. The FFTs uses cues to decide between a gamble pair, where each gamble consists of two fractals.

The are three main components:

* **Feature functions**: Define measurable properties of a gamble pair.
* **Cue class**: Represents a cue based on a feature.
* **FFT class**: Implements sequential decision-making using cues.

---

## Cue Class

### Purpose

The `Cue` class represents a single cue that evaluates two gambles and determines:

* A **cue value** (whether the cue is present or not, and if numerical, the strength of the cue)
* A **preferred side** (left or right, if cue is present)

---

### Mathematical Definition

Given two gambles (G_1) and (G_2), and a feature function (f), define:

$$
F = f(G_1, G_2) - f(G_2, G_1)
$$

$$
\text{Cue Value} = |F|
$$

Decision rule:

If (|F| > \theta):

* left if (F > \theta)
* right if (F < -\theta)

Otherwise:

* No decision (None)

---

### Parameters

* `id`: Unique identifier
* `name`: Short name
* `description`: Text description
* `feature`: Callable function
* `type`: `boolean` or `numerical`
* `threshold`:

  * Boolean: (\theta = 0)
  * Numerical: user-defined
* `params`: Additional parameters for feature function
* `required_args`: Ordered list of inputs

---

### Methods

#### `evaluate`

Evaluates a cue for a single gamble pair (that is, it determines whether the cue is present and if numerical, the strength of the cue)

**Inputs:**

* (x_{\text{left,up}}, x_{\text{left,down}}, x_{\text{right,up}}, x_{\text{right,down}})
* Additional keyword arguments

**Returns:**

* Cue value
* Preferred side (`left`, `right`, or `None`)

---

#### `evaluate_df`

Applies the cue evaluation row-wise to a pandas DataFrame.

**Output columns:**

* `<cue_id>_value`
* `<cue_id>_side_if_true`

---

#### `save_registry`

Saves all instantiated cues to a YAML file.

---

## Feature Functions

Feature functions define how gamble cues are evaluated.

---

### Growth Rate

$$
f(G) =
\begin{cases}
\frac{\log(g_{\text{up}}) + \log(g_{\text{down}})}{2} & \text{multiplicative} \
\frac{g_{\text{up}} + g_{\text{down}}}{2} & \text{additive}
\end{cases}
$$

---

### Expected Isoelastic Utility

For wealth (w) and risk parameter (\eta):

$$
x_1 =
\begin{cases}
g_{\text{up}} \cdot w & \text{multiplicative} \
g_{\text{up}} + w & \text{additive}
\end{cases}
$$

$$
x_2 =
\begin{cases}
g_{\text{down}} \cdot w & \text{multiplicative} \
g_{\text{down}} + w & \text{additive}
\end{cases}
$$

Utility:

$$
U =
\begin{cases}
\frac{\log(x_1) + \log(x_2)}{2} & \eta = 1 \
\frac{x_1^{1-\eta} + x_2^{1-\eta}}{2(1-\eta)} & \eta \neq 1
\end{cases}
$$

---

## FFT Class

### Purpose

The `FFT` (Fast-and-Frugal Tree) class implements sequential decision-making using an ordered list of cues.

Each cue is evaluated in order until a decision is made.
If no decision is made by the end of the tree, the decision is random.
---

### Structure

* `id`: Unique identifier
* `name`: Name of the tree
* `description`: Description
* `cues`: Ordered list of Cue objects

---

### Decision Process

For cues (C_1, C_2, \dots, C_n):

1. Evaluate (C_i)
2. If (C_i) produces a decision:

   * Return decision
3. Otherwise continue

If no cue decides:

* Decision = random choice (left/right)

---

### Methods

#### `decide`

Evaluates cues sequentially for a single input.

**Returns:**

* List of cue values
* Final decision
* Number of cues used

---

#### `decide_df`

Applies FFT to a DataFrame.

**Adds columns:**

* `<cue_id>_cue_value` (for each cue)
* `<fft_id>_decision`
* `<fft_id>_cues_used`

---

#### `save_registry`

Exports all FFT instances to YAML.

---

## Typical Workflow

1. Define feature functions
2. Instantiate Cue objects
3. Build an FFT with ordered cues
4. Evaluate:

   * Single gamble pair using `decide`
   * Dataset using `decide_df`

