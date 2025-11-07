from __future__ import annotations

from copy import deepcopy
from enum import Enum

from .. import logger


class Parameters:
    """
    Factory class for creating parameter dictionaries for different imputation methods.

    Provides static methods to generate properly formatted parameter dictionaries
    with validation and default values for each imputation approach.
    """

    #   List of acceptable input parameters
    #       Commented out if not yet implemented
    class RegressionModel(Enum):
        """
        OLS = 0
        # Probit = 1
        Logit = 2
        # TwoSampleRegression = 3
        """

        OLS = 0
        # Probit = 1
        Logit = 2
        # TwoSampleRegression = 3

    class ErrorDraw(Enum):
        """
        Random = 0
        pmm = 1
        """

        Random = 0
        pmm = 1
        #   rif_binned = 1
        #   rif_pmm = 2

    @staticmethod
    def pmm(
        knearest: int = 10,
        model: RegressionModel = RegressionModel.OLS,
        donate_list: list[str] = None,
        winsor: tuple[float, float] = [0, 1],
        share_leave_out: float = 0.0,
        donate_by: list[str] | str | None = None,
    ) -> dict:
        """
        Parameters for predictive mean matching imputation.

        Parameters
        ----------
        knearest : int, optional
            Number of nearest neighbors for matching, by default 10
        model : RegressionModel, optional
            Regression model type, by default RegressionModel.OLS
        donate_list : list[str], optional
            Additional variables to impute together, by default None
            i.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        winsor : tuple[float, float], optional
            Winsorization percentiles, by default [0, 1]
        share_leave_out : float, optional
            Leave out a share to adjust the predictions to make sure in a random
            sample they match the training sample yhat distribution
            (potentially relevant for LightGBM), by default 0.0
        donate_by : list[str] | str | None, optional
            Grouping variables for donation, by default None

        Returns
        -------
        dict
            PMM parameter dictionary
        """

        if donate_by is None:
            donate_by = []
        elif type(donate_by) is str:
            donate_by = [donate_by]

        if donate_list is None:
            donate_list = []
        return deepcopy(locals())

    @staticmethod
    def LightGBM(
        tune: bool = False,
        tune_overwrite: bool = False,
        parameters: dict | None = None,
        tuner=None,
        tune_hyperparameter_path: str = "",
        quantiles: list = None,
        error: Parameters.ErrorDraw = ErrorDraw.pmm,
        parameters_pmm: dict | None = None,
    ) -> dict:
        """
        Parameters for LightGBM-based imputation.

        Parameters
        ----------
        tune : bool, optional
            Whether to tune hyperparameters, by default False
        tune_overwrite : bool, optional
            Overwrite existing tuned parameters, by default False
        parameters : dict | None, optional
            LightGBM model parameters, by default None
            From NEWS.CodeUtilities.Python.LightGBM
        tuner : Tuner, optional
            Hyperparameter tuner object, by default None
            From NEWS.CodeUtilities.Python.LightGBM
        tune_hyperparameter_path : str, optional
            Path for saving/loading tuned parameters, by default ""
        quantiles : list, optional
            Quantiles for quantile regression, by default None
        error : Parameters.ErrorDraw, optional
            How to convert yhat from LGBM into imputes.
            If pmm, draw from nearest yhat neighbors, for example.
            The default is ErrorDraw.pmm.
        parameters_pmm : dict | None, optional
            PMM parameters if using PMM error drawing, by default None

        Returns
        -------
        dict
            LightGBM parameter dictionary

        Raises
        ------
        Exception
            If quantiles are not between 0 and 1
        """

        if quantiles is None:
            quantiles = []

        if any([qi >= 1 for qi in quantiles]):
            message = f"LightGBM quantiles must be between 0 and 1 (passed {quantiles})"
            logger.error(message)
            raise Exception(message)

        if tuner is None:
            #   Can't tune without a tuner!
            tune = False

        params = deepcopy(locals())
        if parameters_pmm is None and error == Parameters.ErrorDraw.pmm:
            parameters_pmm = Parameters.pmm()

        for keyi, valuei in parameters_pmm.items():
            params[keyi] = valuei

        del params["parameters_pmm"]
        return params

    @staticmethod
    def HotDeck(
        model_list: list[str] | list[list[str]] = None,
        donate_list: list | None = None,
        n_hotdeck_array: int = 3,
        sequential_drop: bool = True,
    ) -> dict:
        """
        Parameters for hot HotDeck imputation

        Parameters
        ----------
        model_list : list[str] | list[list[str]]
            Each model is a list of variables that are used as match keys.
            model_list can either be a list of strings (the model itself)
            or it can be a list of lists of strings (sequential hot deck to match on)
        donate_list : list, optional
            Additional variables to impute together, by default None
            I.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        n_hotdeck_array : int, optional
            Size of hot deck donor arrays, by default 3
        sequential_drop : bool, optional
            Drop variables sequentially until matches found, by default True
                If model_list is a list of strings (one model), should we
                sequentially drop the last variable until all recipients find
                a donor?  Makes it easier to set the hot deck/stat match up.

        Returns
        -------
        dict
            Hot deck parameter dictionary
        """
        #   Pending implementation (if I get to) for deterministic hot decks like the ACS
        #   sort_by:list=None):

        if donate_list is None:
            donate_list = []

        #   Placeholder if I ever want to implement an ACS-style sorted
        #       deterministic hot deck
        sort_by = None
        arguments = deepcopy(locals())

        if sequential_drop:
            #   Create a list of models that sequentially drops the last item
            seq_list = []
            for endi in range(len(model_list), 0, -1):
                if len(model_list[0:endi]) > 0:
                    seq_list.append(model_list[0:endi])

            arguments["model_list"] = seq_list

        return arguments

    @staticmethod
    def StatMatch(
        model_list: list = None, donate_list: list = None, sequential_drop: bool = False
    ):
        """
        Parameters for hot statistical match imputation

        Stat match and hot deck are basically the same, but the hot deck
            iterates over the data carrying arrays of possible donor values
            whereas the stat match just does a join of donors and recipients

        Parameters
        ----------
        model_list : list[str] | list[list[str]]
            Each model is a list of variables that are used as match keys.
            model_list can either be a list of strings (the model itself)
            or it can be a list of lists of strings (sequential hot deck to match on)
        donate_list : list, optional
            Additional variables to impute together, by default None
            I.e., you can predict earnings amount to find a donor,
            but then also impute hours worked and weeks worked with it.
        sequential_drop : bool, optional
            Drop variables sequentially until matches found, by default False
                If model_list is a list of strings (one model), should we
                sequentially drop the last variable until all recipients find
                a donor?  Makes it easier to set the hot deck/stat match up.

        Returns
        -------
        dict
            stat match parameter dictionary
        """
        return Parameters.HotDeck(**locals())

    @staticmethod
    def NearestNeighbor(match_to: str | list[str], parameters_pmm: dict = None) -> dict:
        """
        Match directly to an x variable (or set) using nearest neighbor matching

        Parameters
        ----------
        match_to : str | list[str]
            Variable or list to match to.
        parameters_pmm : dict, optional
            Nearest neighbor match parameters

        Returns
        -------
        dict
            nearest neighbor parameter dictionary
        """
        params = {"match_to": match_to}

        if parameters_pmm is None:
            parameters_pmm = Parameters.pmm()

        for keyi, valuei in parameters_pmm.items():
            params[keyi] = valuei

        return params

    @staticmethod
    def Regression(
        model: RegressionModel = RegressionModel.OLS,
        error: ErrorDraw = ErrorDraw.pmm,
        random_share: float = 1.0,
        parameters_pmm: dict = None,
    ) -> dict:
        """
        Parameters for regression-based imputation.

        Parameters
        ----------
        model : RegressionModel, optional
            Type of regression model, by default RegressionModel.OLS
        error : ErrorDraw, optional
            Method for drawing errors, by default ErrorDraw.Random
            If pmm, draw from nearest yhat neighbors, for example.
            If Random, draw from observed errors for modeled observations
        random_share : float, optional
            Fraction of data to use for regression, by default 1.0
            Use less memory by running the regression on a random subset?
        parameters_pmm : dict, optional
            PMM parameters if using PMM error drawing, by default None

        Returns
        -------
        dict
            Regression parameter dictionary
        """

        params = deepcopy(locals())
        if parameters_pmm is None:
            if error == Parameters.ErrorDraw.pmm:
                parameters_pmm = Parameters.pmm()
            else:
                parameters_pmm = {}

        for keyi, valuei in parameters_pmm.items():
            if keyi != "model":
                params[keyi] = valuei

        del params["parameters_pmm"]
        return params

    # @staticmethod
    # def TwoSampleRegression(parameters_regression:dict|None=None,
    #                         is_boolean:bool=False,
    #                         bins:int=10,
    #                         bin_by:list[str] | None=None,
    #                         percentile_cuts:list[float]|None=None,
    #                         save_percentile_cuts:bool=False,
    #                         round_impute_var_digits:int=4,
    #                         continuous_qtiles_y_cuts:int | list[float] | None=None,
    #                         continuous_qtiles_interpolate_by_bin:bool=False,
    #                         #   cond_match_bins:list[float]|None=None,
    #                         min_n_x_var:int=0,
    #                         draw_error:bool=False,
    #                         path_save:str="",
    #                         path_load:str="",
    #                         load_from_save:bool=False) -> dict:
    #     """
    #     Parameters for two-sample regression imputation.

    #     Parameters
    #     ----------
    #     parameters_regression : dict | None, optional
    #         Underlying regression parameters, by default None
    #         (default for Parameters.Regression()).
    #     is_boolean : bool, optional
    #         Whether variable is binary, by default False
    #     bins : int, optional
    #         Number of prediction bins, by default 10
    #         Separate the yhat into


# 				bins and get the actual
# 				p(y = 1|yhat) or impute for E(y|yhat) in bin
# 				to get reasonable impute values
# 				from the LPM. The default is 10.
#     bin_by : list[str] | None, optional
#         Variables for creating separate bins, by default None
#             i.e. have a different expected value and draws by age or something
#             The ddefault is None (ignore)
#     percentile_cuts : list[float] | None, optional
#         Custom percentile cut points, by default None
#     save_percentile_cuts : bool, optional
#         Whether to save cut point information, by default False
#     round_impute_var_digits : int, optional
#         Decimal places for rounding, by default 4
#             For the cut endpoints and y cut quantiles to
#             (for disclosure, generall)
#     continuous_qtiles_y_cuts : int | list[float] | None, optional
#         For continuous variables, what quantiles to run a quantile
#             regression on to approximate the distribution of values
#             to draw from.
#             Default is [0.1,0.25,0.5,0.75,0.9] if None is passed
#     continuous_qtiles_interpolate_by_bin : bool, optional
#         Interpolate quantiles within bins, by default False
#             If the interpolation range is too wide, it can cause problems
#             Set the interpolation range by bin to ensure coverage (at the cost of time)
#     min_n_x_var : int, optional
#         Minimum observations required per predictor, by default 0
#             For disclosure
#     draw_error : bool, optional
#         Draw errors rather than values, by default False
#     path_save : str, optional
#         Path for saving results, by default ""
#     path_load : str, optional
#         Path for loading existing results, by default ""
#     load_from_save : bool, optional
#         Use saved results instead of re-running, by default False

#     Returns
#     -------
#     dict
#         Two-sample regression parameter dictionary
#     """

#     # if additional_match_vars is None:
#     #     additional_match_vars = []

#     # if cond_match_vars is None:
#     #     cond_match_vars = []

#     if parameters_regression is None:
#         parameters_regression = Parameters.Regression()

#     if parameters_regression["model"] == Parameters.RegressionModel.Probit:
#         message = "Probit model not implemented, use Logit"
#         logger.error(message)
#         raise Exception(message)
#     if percentile_cuts is None:
#         percentile_cuts = []


#     if bin_by is None:
#         bin_by = []
#     # if cond_match_bins is None:
#     #     cond_match_bins = []

#     if continuous_qtiles_y_cuts is None:
#         continuous_qtiles_y_cuts = [0.1,0.25,0.5,0.75,0.9]

#     params = deepcopy(locals())
#     if parameters_regression is None:
#         parameters_regression = Parameters.Regression()

#     for keyi, valuei in parameters_regression.items():
#         params[keyi] = valuei

#     del params["parameters_regression"]
#     return params
