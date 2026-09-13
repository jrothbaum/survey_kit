# Convergence Diagnostics

SRMI is iterative - each variable's imputation model uses the other variables' most recently
imputed values as predictors, and re-runs over several iterations so those relationships can
settle down. Convergence diagnostics answer one question: did that settling-down actually
happen, or is the chain still drifting?

`SRMI.convergence()` returns the numbers behind this - for each variable and iteration:

- **ac** - the lag-1 autocorrelation of the imputed values across implicates. Close to 0 is
  good; it means one iteration's imputed values aren't strongly predictable from the previous
  iteration's.
- **psrf** - the potential scale reduction factor, comparing the spread of values *within* each
  implicate's chain to the spread *across* implicates. Close to 1 is good; well above 1 means
  the implicates haven't mixed - they're still telling meaningfully different stories.

`SRMI.plot_convergence()` gives the visual version: one trace line per implicate, faceted by
variable. Lines that intermingle with no visible trend mean it converged; lines that are still
trending in one direction, or clearly separated from each other, mean it hasn't yet - try more
iterations.

## Example

<iframe src="../../../tutorials/srmi/diagnostics/convergence_trace.html"
    style="width: 100%; height: 700px; border: none;">
</iframe>

## Walkthrough

=== "Code"
    ```python
    --8<-- "tutorials/srmi/diagnostics_convergence.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/srmi/diagnostics_convergence.html){:target="_blank"}
    <iframe src="../../../tutorials/srmi/diagnostics_convergence.html"
        style="width: 100%; height: 800px; border: none;">
    </iframe>
