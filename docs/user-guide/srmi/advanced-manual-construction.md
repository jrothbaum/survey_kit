# Advanced/Manual Construction

`SRMI.simple_model()` covers the common cases, but it's a thin convenience layer over building
`Variable` and `SRMI` objects directly - see [`API`](../../api/srmi.md). Building them by hand
gives you the full range of what the package can do: mixing model families across variables,
custom pre/post-processing functions run at any point in the imputation sequence, hyperparameter
tuning, quantile regression, and anything else not covered by `simple_model()`'s own options.

The tutorials below build everything by hand, one model family at a time.

=== "Hot Deck/Statistical Match"
    Stat match uses a join and hot deck fill forward from an array across the file, but there is no real difference between them theoretically 

    === "Code"
        ```python
        --8<-- "tutorials/srmi/hotdeck.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/hotdeck.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/hotdeck.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Regression"
    Logit and/or OLS-based imputation

    === "Code"
        ```python
        --8<-- "tutorials/srmi/regression.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/regression.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/regression.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Machine Learning (LightGBM)"
    Imputation with LightGBM, see the [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/) for additional information on some of the options.

    === "Code"
        ```python
        --8<-- "tutorials/srmi/gbm.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/gbm.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/gbm.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>

=== "Tabular ML (RandomForest/XGBoost/CatBoost/Multinomial)"
    RandomForest, XGBoost, CatBoost, and bring-your-own sklearn-compatible estimators, plus
    `Multinomial` for an unordered categorical outcome imputed by donor matching rather than a
    predicted value.

    === "Code"
        ```python
        --8<-- "tutorials/srmi/tabular_ml.py"
        ```

    === "Log"
        [View in separate window](../../tutorials/srmi/tabular_ml.html){:target="_blank"}
        <iframe src="../../tutorials/srmi/tabular_ml.html" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>
