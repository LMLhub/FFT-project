# Paper outline

## Main story:
The Copenhagen experiments indicate that humans maximise the time-average growth rate. This means that a typical subject will behave as if they have one utility function under additive dynamics and another under multiplicative dynamics. We want to get a better understanding about how they actually implement the growth rate optimal strategy. Perhaps fast-and-frugal trees (FFT) can provide an explanation.

We have implemented artificial agents with different decision heuristics based on cues, and analysed how accurately they match experimental outcomes, and how close to growth rate optimality they are. The priority heuristic, sum-of-ranks and sign-based heuristics get close to the theoretically optimal result for both dynamics. But we have not found any heuristic that explains choices strictly better than time-average growth rate optimality.

## Outline

- Introduction
    - Background: 
        The CPH experiment
        Figure: Screenshot of one choice from experiment. Link to online version.
        Figure: Individual trajectories
        Time-average growth rate optimality
    - Heuristics: 
        Heuristics as a plausible method for making complex decisions
        FFTs as a class of heuristics
        Figure: Example of FFT (Ratin)
- Method
    - Cues and decision rules
        Construction of cues and decision trees, incl. symmetry issues
        Full list of cue definitions + corresponding trees
        Table: Cues (Ratin)
        Table: FFTs (Ratin)
    - Accuracy measures
        Regular accuracy
        Importance-weighted accuracy
        Accuracy with respect to optimal and actual choices
    - Criteria for selecting and evaluating cues
        Frugality-accuracy tradeoff
        Correct eta (profile rather than estimation dues to "well-known" issues)
- Results
    - Figure: Frugality-accuracy trade-off (Colm)
    - Figure: Accuracy against choices vs accuracy against growth-rate optimality (Emilie)
    - Figure: Importance-weighted accuracy (Emilie)
    - Figure: eta-profiles (Emilie)
- Discussion
    - Did we select the right cues?
    - Did the linearity in ranking impact results?
    - Maybe we actually do have intuition for transformations
- Appendix
    - Method for setting the tolerance for the PH
        Figure: optimal tolerance (Emilie)
