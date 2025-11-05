# Calibration

## What Is It

[Calibration](https://www.jstor.org/stable/2290268) (often implemented via [raking](https://en.wikipedia.org/wiki/Iterative_proportional_fitting)) is used to adjust weights in a sample to make them representative of some other population of interest.  That can be for a survey that's meant to represent the population of a place or a treatment sample that should be representative of some larger group or control group.  

## Why Use It

Calibration can help:

* **Address frame bias** - A group is overrepresented in the frame, such as if we accidentally oversampled a group with a certain characteristic (e.g., high income households)
* **Address nonresponse bias** - Different groups respond at different rates and the results would be biased without accounting for it, as in [this example](https://www.census.gov/newsroom/blogs/research-matters/2025/09/administrative-data-nonresponse-bias-cps-asec.html) (or any critique of polling, maybe)
* **Increase precision** - By using auxiliary information to reduce variance, see [Little and Vartivarian (2005)](https://www150.statcan.gc.ca/n1/en/pub/12-001-x/2005002/article/9046-eng.pdf)
* **Make a sample representative of another population** - Reweight to match a target distribution from a different sample, see [Hainmueller (2012)](https://www.jstor.org/stable/41403737)


### Implementation

This package uses Carl Sanders's [Accelerated Entropy Balancing package](https://github.com/uscensusbureau/entropy-balance-weighting) to implement calibration via entropy balancing. This implementation has proven to be both faster and more robust (converges reliably or can produce "best possible" weights when exact convergence isn't achievable) than other available tools (at least in my anecdotal experience).

**Key advantages:**
- Handles large datasets efficiently
- Robust convergence even with challenging constraints
- Supports bounded weights for practical applications where convergence isn't possible (i.e. slightly conflicting constraints)
 
## API

See the full [Calibration API documentation](/api/calibration) 

## Example/Tutorial

=== "Code"
    ```python
    --8<-- "tutorials/calibration.py"
    ```

=== "Log"
    [View in separate window](/tutorials/calibration.html){:target="_blank"}
    <iframe src="/tutorials/calibration.html" 
        style="width: 100%; height: 800px; border: none;">
    </iframe>