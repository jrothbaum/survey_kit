# Calibration

## What Is It

[Calibration](https://www.jstor.org/stable/2290268) (often implemented via [raking](https://en.wikipedia.org/wiki/Iterative_proportional_fitting)) is used to adjust weights in a sample to make them representative of some other population of interest.  That can be for a survey that's meant to represent the population of a place or a treatment sample that should be representative of some larger group or control group.

The basic idea is to define some set of "moments" that you want your sample to match (such as population controls, share of individuals with certain education levels, share with income in certain bins, etc.) and then reweight your sample so that it matches each of those moments.  See the [Accelerated Entropy Balancing](https://github.com/uscensusbureau/entropy-balance-weighting?tab=readme-ov-file#problem-and-usage) repository for more details on how it works.

## Why Use It

Calibration can help:

* **Address frame bias** - A group is overrepresented in the frame, such as if we accidentally oversampled a group with a certain characteristic (e.g., high income households)
* **Address nonresponse bias** - Different groups respond at different rates and the results would be biased without accounting for it, as in [this example](https://www.census.gov/newsroom/blogs/research-matters/2025/09/administrative-data-nonresponse-bias-cps-asec.html) (or any critique of polling, maybe)
* **Increase precision** - By using auxiliary information to reduce variance, see [Little and Vartivarian (2005)](https://www150.statcan.gc.ca/n1/en/pub/12-001-x/2005002/article/9046-eng.pdf)
* **For causal estimates when comparing treatment and control groups** - Reweight a group (treatment) to match the characteristics of a different group (control), see [Hainmueller (2012)](https://www.jstor.org/stable/41403737)


### Implementation

This package uses [Carl Sanders's](https://sites.google.com/site/carlesanders) [Accelerated Entropy Balancing package](https://github.com/uscensusbureau/entropy-balance-weighting) to implement calibration via entropy balancing. This implementation is both faster and more robust (converges reliably or can produce "best possible" weights when exact convergence isn't achievable) than other available tools (at least in my anecdotal experience).

**Key advantages:**

1. Handles large datasets efficiently
1. Robust convergence even with challenging constraints.
1. Supports bounded weights for practical applications where convergence isn't possible (i.e. slightly conflicting constraints)
 
Advantages 2. and 3. can be incredibly important in practice.  Many surveys weight to state x race x age x gender cells (or something like that).  Let's say we have 3 race/ethnicity groups and 5 age bins, that's 1,530 moment constraints (51 (50 states + DC) x 3 x 5 x 2).  If you start adding in other things (race x income bins, or race x county, or income x county), the number of moments can grow even larger.  

Plus, if you're doing replicate weights, you have to repeat this many times (for example, [160 times in the CPS ASEC](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/2025_ASEC_Replicate_Weight_Usage_Instructions.docx)).  We've found other tools to be impractically slow at scale and/or to have convergence issues (i.e. it just doesn't work and you don't get a weight at the end).

## API

See the full [Calibration API documentation](../api/calibration.md) 

## Example/Tutorial

=== "Code"
    ```python
    --8<-- "tutorials/calibration.py"
    ```

=== "Log"
    [View in separate window](../../tutorials/calibration.html){:target="_blank"}
    <iframe src="../../tutorials/calibration.html" 
        style="width: 100%; height: 800px; border: none;">
    </iframe>