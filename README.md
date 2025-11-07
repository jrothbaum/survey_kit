# Survey Kit

Tools for addressing missing data problems (nonresponse bias and item missingness) including extremely fast calibration weighting and machine learning-based imputation.

A furlough project inspired by the code used for the U.S. Census Bureau for the [National Experimental Wellbeing Statistics (NEWS)](https://www.census.gov/data/experimental-data-products/national-experimental-wellbeing-statistics.html) project.

## Installation
```bash
pip install survey-kit
```

## Features

- **Calibration Weighting** - Fast entropy balancing for nonresponse bias
- **SRMI Imputation** - ML-based multiple imputation with checkpointing
- **Statistics & Standard Errors** - Proper variance estimation for complex surveys

Works with Polars, Pandas, Arrow, and DuckDB. Optimized for large datasets (100K+ rows).

## Documentation

Full documentation: [https://jrothbaum.github.io/survey_kit/](https://jrothbaum.github.io/survey_kit/)

- [Calibration Guide](https://jrothbaum.github.io/survey_kit/user-guide/calibration/)
- [Imputation Guide](https://jrothbaum.github.io/survey_kit/user-guide/imputation/)
- [Statistics Guide](https://jrothbaum.github.io/survey_kit/user-guide/statistics/)

## Support

- [Issues](https://github.com/jrothbaum/survey_kit/issues)
- [Discussions](https://github.com/jrothbaum/survey_kit/discussions)

## License

This project is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the CC0 1.0 Universal public domain dedication.