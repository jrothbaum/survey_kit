from __future__ import annotations
from typing import Callable

import narwhals as nw
from narwhals.typing import IntoFrameT
import re
from survey_kit_formula import ModelSpec
from survey_kit_formula.parser import parse_formula
from survey_kit_formula.terms.classify import referenced_columns

from .inputs import list_input
from .dataframe import (
    columns_from_list,
    _columns_original_order,
    NarwhalsType,
    safe_height,
)

from .. import logger


def _parse_normalized(formula: str):
    """
    Parse a formula with survey_kit_formula's parse_formula(), which requires a
    "~" - callers throughout FormulaBuilder routinely pass rhs-only
    fragments (formulaic tolerated this), so prepend "~" when missing.
    """
    if "~" not in formula:
        formula = f"~{formula}"
    return parse_formula(formula)


def get_model_frame(formula: str, df: IntoFrameT) -> IntoFrameT:
    """
    Build a named model/design matrix from a formula against df - the
    practical `model.matrix(formula, data)` equivalent (R's model.matrix()
    always carries column names, same as this). survey_kit_formula's ModelSpec
    is polars-only, so df is converted to polars around the call and the
    result converted back to df's original backend.
    """
    nw_type = NarwhalsType(df)
    df_polars = nw_type.to_polars()
    return nw_type.from_polars(
        ModelSpec.from_formula(formula, df_polars).get_model_frame(df_polars)
    )


class FormulaBuilder:
    """
    Build and manipulate R-style formulas for statistical models.

    FormulaBuilder provides a programmatic way to construct complex model formulas
    using R/Patsy-style syntax. It supports formula manipulation, variable expansion,
    interactions, transformations, and pattern matching against dataframes.

    The class works with survey_kit_formula to parse and expand formulas, and
    integrates with the dataframe utilities to resolve wildcards and column
    patterns.

    Parameters
    ----------
    df : IntoFrameT | None, optional
        Reference dataframe for resolving column names and wildcards.
        Default is None.
    formula : str, optional
        Initial formula string. If empty, constructs from lhs and constant.
        Default is "".
    lhs : str, optional
        Left-hand side of formula (response variable). Default is "".
    constant : bool, optional
        Include intercept term (1) or suppress it (0). Default is True.

    Attributes
    ----------
    formula : str
        Current formula string.
    df : IntoFrameT | None
        Reference dataframe.
    columns : list[str]
        All variables required by the formula.
    columns_rhs : list[str]
        Variables in right-hand side of formula.
    columns_lhs : list[str]
        Variables in left-hand side of formula.

    Examples
    --------
    Basic formula construction:

    >>> from survey_kit.utilities.formula import FormulaBuilder
    >>>
    >>> fb = FormulaBuilder(df=df, lhs="income", constant=True)
    >>> fb += "age + education"
    >>> print(fb.formula)
    'income~1+age+education'

    Add variables using wildcards:

    >>> fb = FormulaBuilder(df=df, lhs="income")
    >>> fb.continuous(columns="demographic_*")
    >>> fb.factor(columns="region")
    >>> print(fb.formula)

    Polynomial terms:

    >>> fb = FormulaBuilder(df=df, lhs="income")
    >>> fb.polynomial(columns="age", degree=3)
    >>> print(fb.formula)
    'income~1+poly(age,degree=3,raw=True)'

    Interactions:

    >>> fb = FormulaBuilder(df=df, lhs="income")
    >>> fb.simple_interaction(columns=["age", "education"], order=2)
    >>> print(fb.formula)
    'income~1+(age+education)^2'

    Standardization:

    >>> fb = FormulaBuilder(df=df, lhs="income")
    >>> fb.scale(columns=["age", "experience"])
    >>> print(fb.formula)
    'income~1+scale(age)+scale(experience)'

    Working with existing formula strings:

    >>> formula = "income~age+education+age:education"
    >>> fb = FormulaBuilder(df=df, formula=formula)
    >>> fb.expand()  # Expand shorthand
    >>> print(fb.rhs())  # Get right-hand side
    'age+education+age:education'
    >>> print(fb.columns)
    ['income', 'age', 'education']

    Notes
    -----
    Formula syntax follows R/Patsy conventions:
    - `~` separates left and right sides
    - `+` adds terms
    - `:` creates interactions
    - `*` creates main effects and interactions: `a*b` = `a+b+a:b`
    - `^n` creates all n-way interactions
    - `I()` for arithmetic operations
    - `C()` for categorical variables
    - Functions like `scale()`, `center()`, `poly()` for transformations

    Wildcards in column specifications:
    - `"income_*"` matches all columns starting with "income_"
    - `["age", "education_*"]` matches age and all education columns

    The FormulaBuilder can be used in two modes:
    1. Object mode: Create instance and chain methods
    2. Static mode: Call methods with self=None for one-off operations

    FormulaBuilder only builds and manipulates the formula string itself -
    it doesn't touch data. To actually expand a finished formula into a
    model/design matrix, pass its string (`str(fb)`) to `get_model_frame`.
    survey_kit_formula's own ModelSpec (which does the real expansion) only
    accepts polars DataFrame/LazyFrame; get_model_frame converts to polars
    internally and converts the result back to whatever backend you passed
    in (pandas, polars, pyarrow, ...), so it works with any narwhals-
    supported backend even though ModelSpec itself doesn't.

    See Also
    --------
    survey_kit_formula.parser.parse_formula : Underlying formula parser
    get_model_frame : Expand a formula string against data of any backend
    """

    def __init__(
        self,
        df: IntoFrameT | None = None,
        formula: str = "",
        lhs: str = "",
        constant: bool = True,
    ):
        if formula == "":
            self.formula = f"{lhs}~{int(constant)}"
        else:
            self.formula = formula

        if df is not None:
            self.df = nw.from_native(df).head(0).to_native()
        else:
            self.df = None

    def __str__(self):
        return self.formula

    def __add__(self, o):
        """Add terms to the formula using + operator."""
        if type(o) is FormulaBuilder:
            o = o.rhs()

        o = str(o)
        if o.startswith("~"):
            o = o[1 : len(o)]

        if o.startswith("1+"):
            o = o[2 : len(o)]
        if o.startswith("0+"):
            o = o[2 : len(o)]

        self.formula = f"{self.formula}+{o}"

        self.expand()

        return self

    def add_to_formula(self, add_part: str = "", plus_first: bool = True) -> None:
        """
        Append a string to the formula.

        Parameters
        ----------
        add_part : str, optional
            String to append. Default is "".
        plus_first : bool, optional
            Add "+" before the string. Default is True.
        """
        if plus_first:
            plus = "+"
        else:
            plus = ""
        self.formula += f"{plus}{add_part}"

    def any_wrapper(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        case_insensitive: bool = False,
        prefix: str = "",
        suffix: str = "",
    ) -> str | FormulaBuilder:
        """
        General wrapper for adding columns with prefix/suffix.

        Used internally by other methods to add variables with transformations.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for resolving column patterns. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause (bypasses column lookup). Default is "".
        case_insensitive : bool, optional
            Case-insensitive column matching. Default is False.
        prefix : str, optional
            String to prepend to each column. Default is "".
        suffix : str, optional
            String to append to each column. Default is "".

        Returns
        -------
        str | FormulaBuilder
            Formula string if self is None, otherwise returns self for chaining.
        """
        columns = list_input(columns)

        #   Dataframe to look for columns in
        if df is None:
            df = self.df

        if clause != "":
            output = f"+{prefix}{clause}{suffix}"
        else:
            if df is not None:
                columns = columns_from_list(
                    df=df, columns=columns, case_insensitive=case_insensitive
                )

            out_list = [f"{prefix}{coli}{suffix}" for coli in columns]
            output = "+" + "+".join(out_list)

        if self is None:
            return output
        else:
            self.formula += output
            return self

    def continuous(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        case_insensitive: bool = False,
    ) -> str | FormulaBuilder:
        """
        Add continuous variables to the formula.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns (e.g., "income_*"). Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb = FormulaBuilder(df=df, lhs="y")
        >>> fb.continuous(columns=["age", "income"])
        >>> print(fb.formula)
        'y~1+age+income'
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        return caller.any_wrapper(
            df=df, columns=columns, clause=clause, case_insensitive=case_insensitive
        )

    def function(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        operator_before: str = "",
        operator_after: str = "",
        function_item: str = "",
        case_insensitive: bool = False,
        **kwargs,
    ) -> str | FormulaBuilder:
        """
        Wrap columns in a function call.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        operator_before : str, optional
            String before function call. Default is "".
        operator_after : str, optional
            String after function call. Default is "".
        function_item : str, optional
            Function name (e.g., "log", "sqrt"). Default is "".
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.
        **kwargs
            Additional function arguments.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.function(columns="income", function_item="log")
        >>> print(fb.formula)
        'y~1+log(income)'
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        if function_item != "":
            operator_before = f"{operator_before}{function_item}("

            operator_after_final = ""
            if len(kwargs):
                for keyi, valuei in kwargs.items():
                    operator_after_final += ","
                    if type(valuei) is str:
                        operator_after_final += f"{keyi}='{valuei}'"
                    else:
                        operator_after_final += f"{keyi}={valuei}"

            operator_after_final += f"{operator_after})"
        else:
            operator_after_final = operator_after

        return caller.any_wrapper(
            df=df,
            columns=columns,
            clause=clause,
            case_insensitive=case_insensitive,
            prefix=f"{{{operator_before}",
            suffix=f"{operator_after_final}}}",
        )

    def scale(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        standardize: bool = True,
        case_insensitive: bool = False,
    ) -> str | FormulaBuilder:
        """
        Standardize or center variables.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        standardize : bool, optional
            If True, standardize (mean=0, sd=1). If False, only center (mean=0).
            Default is True.
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.scale(columns="age")  # Standardize
        >>> fb.scale(columns="income", standardize=False)  # Center only
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        if standardize:
            function_item = "scale"
        else:
            function_item = "center"

        return caller.any_wrapper(
            df=df,
            columns=columns,
            clause=clause,
            case_insensitive=case_insensitive,
            prefix=f"{function_item}(",
            suffix=")",
        )

    def center(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        case_insensitive: bool = False,
    ) -> str | FormulaBuilder:
        """
        Center variables (subtract mean).

        Convenience method for scale(standardize=False).

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        return caller.scale(
            df=df,
            columns=columns,
            clause=clause,
            case_insensitive=case_insensitive,
            standardize=False,
        )

    def standardize(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        case_insensitive: bool = False,
    ) -> str | FormulaBuilder:
        """
        Standardize variables (mean=0, sd=1).

        Convenience method for scale(standardize=True).

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        return caller.scale(
            df=df,
            columns=columns,
            clause=clause,
            case_insensitive=case_insensitive,
            standardize=True,
        )

    def polynomial(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        degree: int = 0,
        case_insensitive: bool = False,
        center: bool = False,
    ) -> str | FormulaBuilder:
        """
        Add polynomial terms.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        degree : int, optional
            Polynomial degree. Default is 0 (returns continuous).
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.
        center : bool, optional
            Use orthogonal polynomials (centered). Default is False (raw polynomials).

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.polynomial(columns="age", degree=3)
        >>> print(fb.formula)
        'y~1+poly(age,degree=3,raw=True)'

        >>> fb.polynomial(columns="age", degree=2, center=True)
        >>> print(fb.formula)
        'y~1+poly(age,degree=2,raw=False)'
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self
            if df is None:
                df = self.df

        if degree <= 1:
            if center:
                return caller.center(
                    df=df,
                    columns=columns,
                    clause=clause,
                    case_insensitive=case_insensitive,
                )
            else:
                return caller.continuous(
                    df=df,
                    columns=columns,
                    clause=clause,
                    case_insensitive=case_insensitive,
                )
        else:
            subformula = ""

            for power in range(1, degree + 1):
                operator_before = "poly("
                if center:
                    operator_after = f",degree={degree},raw=False)"
                else:
                    operator_after = f",degree={degree},raw=True)"

                subformula += FormulaBuilder.function(
                    df=df,
                    columns=columns,
                    clause=clause,
                    operator_before=operator_before,
                    operator_after=operator_after,
                    case_insensitive=case_insensitive,
                )

            if self is None:
                return subformula
            else:
                self.add_to_formula(subformula, False)
                return self

    def simple_interaction(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        order: int = 2,
        case_insensitive: bool = False,
        sub_function: Callable | None = None,
        no_base: bool = False,
    ) -> str | FormulaBuilder:
        """
        Create interactions between variables.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        order : int, optional
            Interaction order. 2 = pairwise, 3 = three-way, etc. Default is 2.
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.
        sub_function : Callable | None, optional
            Function to apply to columns before interacting. Default is None.
        no_base : bool, optional
            Exclude main effects (only interactions). Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.simple_interaction(columns=["age", "education"], order=2)
        >>> print(fb.formula)
        'y~1+(age+education)^2'

        >>> fb.simple_interaction(columns=["a", "b", "c"], order=2, no_base=True)
        >>> print(fb.formula)
        'y~1+(a+b+c)^2-(a+b+c)'  # Interactions only
        """
        if self is not None:
            if df is None:
                df = self.df

        if sub_function is None:
            sub_function = FormulaBuilder.continuous

        columns = columns_from_list(
            df=df, columns=columns, case_insensitive=case_insensitive
        )

        subformula = sub_function(
            df=df, columns=columns, case_insensitive=case_insensitive
        )
        #   Remove leading plus sign
        subformula = subformula[1 : len(subformula)]

        if len(columns) > 1:
            #   R's actual "all interactions up to order N" operator is "^",
            #   not the "**" formulaic tolerated as an alias.
            output = f"({subformula})^{order}"
        else:
            output = subformula

        if no_base:
            output += f"-({subformula})"

        if self is None:
            return f"+{output}"
        else:
            self.add_to_formula(output)
            return self

    def interact_clauses(
        self=None, clause1: str = "", clause2: str = "", no_base: bool = False
    ) -> str | FormulaBuilder:
        """
        Create interactions between two formula clauses.

        Parameters
        ----------
        clause1 : str, optional
            First formula clause. Default is "".
        clause2 : str, optional
            Second formula clause. Default is "".
        no_base : bool, optional
            Exclude main effects from clauses. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.interact_clauses("age+education", "region")
        >>> print(fb.formula)
        'y~1+(age+education)*(region)'
        """
        if clause1.startswith("+"):
            clause1 = clause1[1:]
        if clause2.startswith("+"):
            clause2 = clause2[1:]

        clause1 = clause1.replace("++", "+")
        clause2 = clause2.replace("++", "+")

        output = f"({clause1})*({clause2})"

        if no_base:
            output += f"-({clause1} + {clause2})"

        if self is None:
            return f"+{output}"
        else:
            self.add_to_formula(output)
            return self

    def factor(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        clause: str = "",
        reference=None,
        case_insensitive: bool = False,
    ):
        """
        Add categorical variables with treatment coding.

        Parameters
        ----------
        df : IntoFrameT | None, optional
            Dataframe for column lookup. Default is None.
        columns : str | list | None, optional
            Column names or patterns. Default is None.
        clause : str, optional
            Pre-constructed clause. Default is "".
        reference : str | int | None, optional
            Reference level for treatment coding. Default is None (use first level).
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Formula string or self for chaining.

        Examples
        --------
        >>> fb.factor(columns="region")
        >>> print(fb.formula)
        'y~1+C(region)'

        >>> fb.factor(columns="education", reference="high_school")
        >>> print(fb.formula)
        'y~1+C(education, contr.treatment, base=2)'
        """
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self
            if df is None:
                df = self.df

        if reference is None:
            return caller.any_wrapper(
                df=df,
                columns=columns,
                clause=clause,
                case_insensitive=case_insensitive,
                prefix="C(",
                suffix=")",
            )

        if type(reference) is not str:
            #   Already a 1-indexed level position, matching R's actual
            #   C(x, contr, ...) convention: the contrast function as a bare
            #   2nd positional arg, "base" forwarded as a separate keyword -
            #   not C(x, contr.treatment(base=N)), which isn't valid R either.
            return caller.any_wrapper(
                df=df,
                columns=columns,
                clause=clause,
                case_insensitive=case_insensitive,
                prefix="C(",
                suffix=f", contr.treatment, base={reference})",
            )

        #   String reference - R's contr.treatment "base" is a 1-indexed
        #   integer position, not a label, so resolve it per-column against
        #   that column's own sorted non-null unique levels (the same
        #   ordering survey_kit_formula's ModelSpec uses to assign levels).
        #   self.df only ever keeps a 0-row schema reference (see __init__),
        #   which has no values to resolve a label against - a real df with
        #   actual rows must be passed explicitly here.
        if df is None or safe_height(df) == 0:
            raise ValueError(
                "factor() needs a dataframe with actual rows (not just "
                "schema) to resolve a string reference level to its "
                "1-indexed position - pass df=<full dataframe> explicitly"
            )

        resolved_columns = columns_from_list(
            df=df, columns=list_input(columns), case_insensitive=case_insensitive
        )

        nw_df = nw.from_native(df).lazy()
        schema = nw_df.collect_schema()

        clauses = []
        for coli in resolved_columns:
            if schema[coli] == nw.Boolean:
                levels = ["False", "True"]
            else:
                levels = (
                    nw_df.select(nw.col(coli).cast(nw.String))
                    .collect()[coli]
                    .drop_nulls()
                    .unique()
                    .sort()
                    .to_list()
                )

            if reference not in levels:
                raise ValueError(
                    f"reference level {reference!r} not found in column "
                    f"{coli!r} (levels: {levels!r})"
                )
            base = levels.index(reference) + 1
            clauses.append(f"C({coli}, contr.treatment, base={base})")

        output = "+" + "+".join(clauses)

        if self is None:
            return output
        else:
            self.formula += output
            return self

    @property
    def columns(self):
        """Get all variables required by the formula."""
        return self.columns_from_formula()

    def columns_from_formula(self=None, formula: str = "") -> list[str]:
        """
        Extract variable names from a formula.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        list[str]
            Variable names required by the formula.
        """
        if type(self) is str:
            formula = self
            self = None

        if self is not None and formula == "":
            formula = self.formula

        parsed = _parse_normalized(formula)

        seen = set()
        out = []

        def _add_var(vari):
            for coli in referenced_columns(vari):
                if coli not in seen:
                    seen.add(coli)
                    out.append(coli)

        if parsed.lhs is not None:
            _add_var(parsed.lhs)

        for termi in parsed.rhs:
            for vari in termi.vars:
                _add_var(vari)

        return out

    def lhs(self=None, formula: str = "") -> str:
        """
        Get left-hand side of formula.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        str
            Left-hand side (response variable).
        """
        if type(self) is str:
            formula = self
            self = None

        if self is not None:
            if formula == "":
                formula = self.formula

        #   Separate into subclauses
        sides = formula.split("~")

        if len(sides) == 2:
            lhs_string = sides[0]
        else:
            lhs_string = ""

        return lhs_string

    def rhs(self=None, formula: str = "") -> str:
        """
        Get right-hand side of formula.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        str
            Right-hand side (predictors).
        """
        if type(self) is str:
            formula = self
            self = None

        if self is not None:
            if formula == "":
                formula = self.formula

        #   Separate into subclauses
        sides = formula.split("~")

        if len(sides) == 2:
            rhs_string = sides[1]
        else:
            rhs_string = sides[0]

        return rhs_string

    @property
    def columns_rhs(self):
        """Variables in right-hand side, ordered as in dataframe."""
        formula_rhs = self.rhs()
        columns = FormulaBuilder.columns_from_formula(formula=formula_rhs)

        if self.df is not None:
            return _columns_original_order(
                columns_unordered=columns,
                columns_ordered=nw.from_native(self.df).lazy().collect_schema().names(),
            )
        else:
            return columns

    @property
    def columns_lhs(self):
        """Variables in left-hand side, ordered as in dataframe."""
        formula_lhs = self.lhs()
        if formula_lhs == "":
            return []
        else:
            columns = FormulaBuilder.columns_from_formula(formula=formula_lhs)
            if len(columns) <= 1:
                return columns

            if self.df is not None:
                return _columns_original_order(
                    columns_unordered=columns,
                    columns_ordered=nw.from_native(self.df)
                    .lazy()
                    .collect_schema()
                    .names(),
                )
            else:
                return columns

    def has_constant(
        self=None, formula: str = "", true_if_missing: bool = False
    ) -> bool:
        """
        Check if formula includes an intercept.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".
        true_if_missing : bool, optional
            Return True if constant term is implicit (not specified).
            Default is False.

        Returns
        -------
        bool
            True if formula includes intercept.
        """
        if type(self) is str:
            formula = self
            self = None

        if self is not None:
            if formula == "":
                formula = self.formula

        #   Separate into subclauses
        sides = formula.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = None
            rhs = sides[0]

        if true_if_missing:
            return not rhs.replace(" ", "").startswith("0")
        else:
            return rhs.replace(" ", "").startswith("1")

    def remove_constant(self):
        """Remove intercept from formula (change ~1+ to ~0+)."""
        self.formula = self.formula.replace("~1+", "~")
        self.formula = self.formula.replace("~0+", "~")
        self.formula = self.formula.replace("~", "~0+")

    def expand(self=None, formula: str = ""):
        """
        Expand formula shorthand (e.g., a*b becomes a+b+a:b).

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        str
            Expanded formula string.
        """
        if self is not None:
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(formula=formula)

        lhs = self.lhs()
        rhs = self.rhs()

        parsed = _parse_normalized(rhs)
        #   survey_kit_formula pulls the intercept out into its own field rather
        #   than keeping it as a literal "1"/"0" term - re-add it explicitly
        #   so the reconstructed formula matches formulaic's prior shape.
        intercept_term = "1" if parsed.intercept else "0"
        reconstructed_formula = "+".join(
            [intercept_term] + [str(termi) for termi in parsed.rhs]
        )

        if lhs != "":
            reconstructed_formula = f"{lhs}~{reconstructed_formula}"

        if self is not None:
            self.formula = reconstructed_formula

        return reconstructed_formula

    def exclude_interactions(
        self=None,
        formula: str = "",
        b_exclude_powers: bool = True,
        df: IntoFrameT | None = None,
    ) -> tuple[str, bool]:
        """
        Remove interaction terms from formula.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".
        b_exclude_powers : bool, optional
            Also exclude polynomial terms. Default is True.
        df : IntoFrameT | None, optional
            Reference dataframe. Default is None.

        Returns
        -------
        tuple[str, bool]
            (modified formula, whether any terms were dropped)
        """
        if self is not None:
            if df is None:
                df = self.df
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(df=df, formula=formula)

        #   It's easier with the expanded formula
        self.expand()

        #   Separate into subclauses
        sides = self.formula.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = ""
            rhs = sides[0]
        subclauses = rhs.split("+")

        rhs = ""

        any_dropped = False

        for clausei in subclauses:
            if b_exclude_powers and "^" in clausei:
                any_dropped = True
                logger.info(
                    f"Dropping {clausei} from formula for having a variable to a power"
                )
            elif ":" in clausei:
                #   Direct interaction
                any_dropped = True
                logger.info(f"Dropping {clausei} from formula")
            else:
                #   include this in the final formula
                rhs += f"+{clausei.strip()}"

        #   Get rid of leading +
        rhs = rhs[1 : len(rhs)]

        output = f"{lhs}~{rhs}"

        self.formula = output
        return (output, any_dropped)

    def exclude_variables(
        self=None,
        exclude_list: list = None,
        formula: str = "",
        df: IntoFrameT | None = None,
        case_insensitive: bool = False,
    ):
        """
        Remove specific variables from formula.

        Parameters
        ----------
        exclude_list : list, optional
            Variables to exclude. Default is None.
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".
        df : IntoFrameT | None, optional
            Reference dataframe. Default is None.
        case_insensitive : bool, optional
            Case-insensitive matching. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Modified formula string or self for chaining.
        """
        if exclude_list is None:
            exclude_list = []

        if self is not None:
            if df is None:
                df = self.df
            if formula == "":
                formula = self.formula

        if df is not None:
            exclude_list = columns_from_list(
                df=df, columns=exclude_list, case_insensitive=case_insensitive
            )

        #   Separate into subclauses
        sides = formula.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = None
            rhs = sides[0]
        subclauses = rhs.split("+")

        rhs = ""

        regex_list = [
            f"(^|([^a-zA-Z0-9_])){itemi}($|([^a-zA-Z0-9_]))" for itemi in exclude_list
        ]
        regexes = "(" + ")|(".join(regex_list) + ")"

        for clausei in subclauses:
            if re.match(regexes, clausei) is None:
                #   include this in the final formula
                rhs += f"+{clausei}"
            else:
                logger.info(f"Dropping {clausei} from formula")

        #   Get rid of leading +
        rhs = rhs[1 : len(rhs)]

        if lhs is not None:
            output = f"{lhs}~{rhs}"
        else:
            output = rhs

        if self is None:
            return output
        else:
            self.formula = output
            return self

    def match_formula_to_columns(
        self=None,
        columns: list[str] | str | None = None,
        formula: str = "",
    ) -> str | FormulaBuilder:
        """
        Restrict a formula's RHS to only the terms that produced at least
        one of the given (expanded) model-matrix column names - the
        inverse of formula expansion. Used after variable selection (e.g.
        LASSO/RFECV) has already chosen a subset of model-matrix columns
        to keep, to turn that back into a formula.

        Column names are matched against a term's own deparsed text as a
        per-":"-part prefix (e.g. term "C(region):age" matches column
        "C(region)b:age"), matching how survey_kit_formula/R build model matrix
        column names: variable prefix (deparsed call or var name) plus a
        per-variable suffix (factor level, poly index, ...), joined by ":"
        for interactions.

        Parameters
        ----------
        columns : list[str] | str | None, optional
            Expanded model-matrix column names to keep. Default is None.
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        str | FormulaBuilder
            Modified formula string or self for chaining.
        """
        columns = list_input(columns)

        if self is not None:
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(formula=formula)

        lhs = self.lhs(formula=formula)
        parsed = _parse_normalized(self.rhs(formula=formula))
        intercept_term = "1" if parsed.intercept else "0"

        kept_terms = [intercept_term]
        for termi in parsed.rhs:
            term_prefixes = str(termi).split(":")

            matched = False
            for coli in columns:
                col_parts = coli.split(":")
                if len(col_parts) != len(term_prefixes):
                    continue
                if all(
                    colp.startswith(prefixp)
                    for colp, prefixp in zip(col_parts, term_prefixes)
                ):
                    matched = True
                    break

            if matched:
                kept_terms.append(str(termi))

        output = "+".join(kept_terms)
        if lhs != "":
            output = f"{lhs}~{output}"

        if self is None:
            return output
        else:
            self.formula = output
            return self

    def add_base_from_interactions(
        self=None, formula: str = ""
    ) -> str | FormulaBuilder:
        """
        Ensure every interaction term's main-effect (single-variable)
        components are also present in the formula, adding any that are
        missing. Enforces the usual marginality/hierarchy convention after
        selection (e.g. match_formula_to_columns) may have kept an
        interaction without its base terms.

        Parameters
        ----------
        formula : str, optional
            Formula string. If empty, uses self.formula. Default is "".

        Returns
        -------
        str | FormulaBuilder
            Modified formula string or self for chaining.
        """
        if self is not None:
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(formula=formula)

        lhs = self.lhs(formula=formula)
        parsed = _parse_normalized(self.rhs(formula=formula))
        intercept_term = "1" if parsed.intercept else "0"

        existing_terms = [str(termi) for termi in parsed.rhs]
        existing_set = set(existing_terms)

        added_terms = []
        for termi in parsed.rhs:
            if termi.order > 1:
                for vari in termi.vars:
                    var_str = str(vari)
                    if var_str not in existing_set:
                        existing_set.add(var_str)
                        added_terms.append(var_str)

        output = "+".join([intercept_term] + existing_terms + added_terms)
        if lhs != "":
            output = f"{lhs}~{output}"

        if self is None:
            return output
        else:
            self.formula = output
            return self

    def formula_with_varnames_in_brackets(
        self=None,
        clause: str = "",
        df: pl.LazyFrame | pl.DataFrame | None = None,
        case_insensitive: bool = False,
        append: bool = False,
    ) -> str | FormulaBuilder:
        """
        Expand {pattern} placeholders with matching column names.

        Replaces {var*} with all columns matching var* pattern in the dataframe.
        Useful for programmatically building formulas with wildcards.

        Parameters
        ----------
        clause : str, optional
            Formula clause with {pattern} placeholders. Default is "".
        df : pl.LazyFrame | pl.DataFrame | None, optional
            Dataframe for column lookup. Default is None.
        case_insensitive : bool, optional
            Case-insensitive pattern matching. Default is False.
        append : bool, optional
            Append to existing formula instead of replacing.
            Only used when called on instance. Default is False.

        Returns
        -------
        str | FormulaBuilder
            Expanded formula string or self for chaining.

        Examples
        --------
        >>> fb = FormulaBuilder(df=df)
        >>> fb.formula_with_varnames_in_brackets("{income_*} + age")
        >>> print(fb.formula)
        'y~1+income_wages+income_self_employment+age'
        """

        call_recursively = False
        if df is None and self is not None:
            df = self.df

        #   Separate into subclauses
        sides = clause.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = None
            rhs = sides[0]
        subclauses = rhs.split("+")

        rhs = ""
        for clausei in subclauses:
            replaced_clause = ""

            left_bracket = clausei.find("{")
            right_bracket = clausei.find("}")

            if left_bracket >= 0 and right_bracket >= 0:
                replace_string = clausei[left_bracket : right_bracket + 1]
                var_name = replace_string[1 : len(replace_string) - 1]
                Columns = columns_from_list(
                    df=df, columns=[var_name], case_insensitive=case_insensitive
                )

                for coli in Columns:
                    if replaced_clause != "":
                        replaced_clause += "+"
                    replaced_clause += clausei.replace(replace_string, coli)

                clausei = replaced_clause

            call_recursively = clausei.find("{") >= 0 and clausei.find("}") >= 0
            rhs += f"+{clausei}"

        #   Get rid of leading +
        rhs = rhs[1 : len(rhs)]

        if lhs is not None:
            output = f"{lhs}~{rhs}"
        else:
            output = rhs

        if call_recursively:
            output = FormulaBuilder.formula_with_varnames_in_brackets(
                clause=output, df=df, case_insensitive=case_insensitive
            )

        if self is None:
            return output
        else:
            if append:
                self.add_to_formula(output)
            else:
                self.formula = output
            return self
