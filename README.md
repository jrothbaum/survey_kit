# Survey Kit

Tools for addressing missing data problems (nonresponse bias and item missingness) including extremely fast calibration weighting and machine learning-based imputation.

A furlough project inspired by the code used for the U.S. Census Bureau for the [National Experimental Wellbeing Statistics (NEWS)](https://www.census.gov/data/experimental-data-products/national-experimental-wellbeing-statistics.html) project.

## Installation
```bash
uv add survey-kit
# or
pip install survey-kit
```

Running regressions with multiply imputed data (below) can be easier with the tools noted below, but they are not added as dependencies by default. Add only what you use:
```bash
uv add "survey-kit[pyfixest]"    # or [statsmodels], [linearmodels], [polars-ds], [r], [stata]
uv add "survey-kit[all-stats]"   # everything at once
```

For R and Stata, there's some additional setup you need to do to get them working with python. See [rpy2's installation instructions](https://rpy2.github.io/doc/latest/html/overview.html) and [pystata's documentation](https://www.stata.com/python/pystata/).

## Features

- **Calibration Weighting** - Fast entropy balancing for nonresponse bias
- **SRMI Imputation** - ML-based multiple imputation with checkpointing
- **Statistics & Standard Errors** - Proper variance estimation for complex surveys
- **Running regressions with multiply imputed data using standard python, R, or Stata tools** - statsmodels, linearmodels, pyfixest, R, and Stata behind one interface

Works with Polars, Pandas, Arrow, and DuckDB.

## Documentation

Full documentation: [https://jrothbaum.github.io/survey_kit/](https://jrothbaum.github.io/survey_kit/)

- [Calibration Guide](https://jrothbaum.github.io/survey_kit/user-guide/calibration/)
- [Imputation Guide](https://jrothbaum.github.io/survey_kit/user-guide/srmi/)
- [Statistics Guide](https://jrothbaum.github.io/survey_kit/user-guide/statistics/)
- [Guide to running regressions with multiply imputed data using standard python, R, or Stata tools](https://jrothbaum.github.io/survey_kit/user-guide/adapters/)

## Support

- [Issues](https://github.com/jrothbaum/survey_kit/issues)

## License

This project is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the CC0 1.0 Universal public domain dedication.