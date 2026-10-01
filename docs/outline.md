# Paper outline

## Main story:
The Copenhagen experiments indicate that humans maximise the time-average growth rate. This means that a typical subject will behave as if they have one utility function under additive dynamics and another under multiplicative dynamics. We want to get a better understanding about how they actually implement the growth rate optimal strategy. Perhaps fast-and-frugal trees (FFT) can provide an explanation.

We have implemented artificial agents with different decision heuristics based on cues, and analysed how accurately they match experimental outcomes, and how close to growth rate optimality they are. The priority heuristic, sum-of-ranks and sign-based heuristics get close to the theoretically optimal result for both dynamics. But we have not found any heuristic that explains choices strictly better than time-average growth rate optimality.

## Outline

- Introduction
    - Background: 
        The CPH experiment
        Time-average growth rate optimality
    - Heuristics: 
        Heuristics as a plausible method for making complex decisions
        FFTs as a class of heuristics
- Method
    - Cues and decision rules
        Construction of cues and decision trees, incl. symmetry issues
        Full list of cue definitions + corresponding trees
    - Accuracy measures
        Regular accuracy
        Importance weighted
        Accuracy with respect to optimal and actual choices
    - Criteria for selecting and evaluating cues
        Frugality-accuracy tradeoff
        Correct eta
- Results
- Discussion
    - Did we select the right cues?
