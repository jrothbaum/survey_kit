from __future__ import annotations
from typing import Callable

import re
import narwhals as nw
from narwhals.typing import IntoFrameT
import polars as pl
from formulaic import Formula
from formulaic.parser import DefaultFormulaParser

from .inputs import list_input
from .dataframe import columns_from_list, _columns_original_order
from .. import logger


class FormulaBuilder:
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

    # #   Simple Interactions within group, where Order # determines interaction depths
    # #       2 = pairwise/second order
    # #       3
    def simple_interaction(
        self=None,
        df: IntoFrameT | None = None,
        columns: str | list | None = None,
        order: int = 2,
        case_insensitive: bool = False,
        sub_function: Callable | None = None,
        no_base: bool = False,
    ) -> str | FormulaBuilder:
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
            output = f"({subformula})**{order}"
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
        if self is None:
            caller = FormulaBuilder
        else:
            caller = self

        prefix = "C("
        if reference is not None:
            if type(reference) is str:
                suffix = f", contr.treatment('{reference}'))"
            else:
                suffix = f", contr.treatment({reference}))"

        else:
            suffix = ")"  #  f", contr.treatment)"

        return caller.any_wrapper(
            df=df, columns=columns, clause=clause, prefix=prefix, suffix=suffix
        )

    @property
    def columns(self):
        return self.columns_from_formula()

    def columns_from_formula(self=None, formula: str = "") -> list[str]:
        if type(self) is str:
            formula = self
            self = None

        if self is not None and formula == "":
            formula = self.formula
        return list(Formula(formula).required_variables)

    def lhs(self=None, formula: str = "") -> str:
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
        self.formula = self.formula.replace("~1+", "~")
        self.formula = self.formula.replace("~0+", "~")
        self.formula = self.formula.replace("~", "~0+")

    def interactions_with_cols_to_list(
        self=None,
        formula: str = "",
        df: IntoFrameT | None = None,
        col_check: list = None,
    ) -> dict:
        interactions_dict = FormulaBuilder.interactions_with_cols_to_dict(
            self=self, formula=formula, df=df, col_check=col_check
        )

        outputs = []
        for valuei in interactions_dict.values():
            outputs.extend(valuei)

        #   Remove duplicates and returns
        return _columns_original_order(
            cols_unordered=list(set(outputs)), cols_ordered=outputs
        )

    def interactions_with_cols_to_dict(
        self=None,
        formula: str = "",
        df: IntoFrameT | None = None,
        col_check: list = None,
    ) -> dict:
        if self is not None:
            if df is None:
                df = self.df
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(df=df, formula=formula)

        self.expand()

        logger.info(self.formula)

        #   Separate into subclauses
        sides = self.formula.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = None
            rhs = sides[0]

        subclauses = rhs.split("+")

        interactions = {coli: [] for coli in col_check}
        for clausei in subclauses:
            fbi = FormulaBuilder(df=df, formula=f"~{clausei}")

            cols_in_clausei = fbi.columns
            for coli in col_check:
                if coli in cols_in_clausei:
                    interactions[coli].append(clausei)

        return interactions

    def add_base_from_interactions(
        self=None, formula: str = "", df: IntoFrameT | None = None
    ):
        if self is not None:
            if df is None:
                df = self.df
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
        subclauses = rhs.split("+")

        rhs = ""

        interactions = []
        non_interactions = []
        for clausei in subclauses:
            if ":" in clausei:
                interactions.append(clausei)
            else:
                non_interactions.append(clausei.strip())

            rhs += f"+{clausei.strip()}"

        for interi in interactions:
            sub_clauses = interi.split(":")

            for subi in sub_clauses:
                subi = subi.strip()
                if subi not in non_interactions:
                    rhs += f"+{subi}"

        #   Get rid of leading +
        rhs = rhs[1 : len(rhs)]

        if lhs is not None:
            output = f"{lhs}~{rhs}"
        else:
            output = rhs

        if self is not None:
            self.formula = output

        return output

    def exclude_interactions(
        self=None,
        formula: str = "",
        b_exclude_powers: bool = True,
        df: IntoFrameT | None = None,
    ) -> tuple[str, bool]:
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

    def expand(self=None, formula: str = ""):
        if self is not None:
            if formula == "":
                formula = self.formula
        else:
            self = FormulaBuilder(formula=formula)

        lhs = self.lhs()
        rhs = self.rhs()

        parser = DefaultFormulaParser()
        parsed = parser.get_terms(rhs)
        reconstructed_formula = "+".join([str(i) for i in parsed])

        if lhs != "":
            reconstructed_formula = f"{lhs}~{reconstructed_formula}"

        if self is not None:
            self.formula = reconstructed_formula

        return reconstructed_formula

    def recode_to_continuous(
        self=None,
        df: pl.LazyFrame | pl.DataFrame | None = None,
        formula: str = "",
        remove_factor: bool = True,
        remove_scale: bool = True,
    ) -> tuple[str, list[str]]:
        if self is not None:
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

        processed_rhs = ""
        recoded_factors = []
        for clausei in subclauses:
            if ":" in clausei:
                #   Interaction
                interaction = ""
                for subi in clausei.split(":"):
                    if interaction != "":
                        colon = ":"
                    else:
                        colon = ""

                    if FormulaBuilder._is_factor(subi):
                        if remove_factor:
                            subi = FormulaBuilder.columns_from_formula(
                                formula=f"~{subi}"
                            )[0]
                            recoded_factors.append(subi)
                    elif FormulaBuilder._is_scale(subi):
                        if remove_scale:
                            subi = FormulaBuilder.columns_from_formula(
                                formula=f"~{subi}"
                            )[0]
                    interaction += f"{colon}{subi}"

                processed_rhs += f"+{interaction}"
            else:
                if FormulaBuilder._is_factor(clausei):
                    if remove_factor:
                        clausei = FormulaBuilder.columns_from_formula(
                            formula=f"~{clausei}"
                        )[0]
                        recoded_factors.append(clausei)
                elif FormulaBuilder._is_scale(clausei):
                    if remove_scale:
                        clausei = FormulaBuilder.columns_from_formula(
                            formula=f"~{clausei}"
                        )[0]

                processed_rhs += f"+{clausei}"

        #   Get rid of leading +
        processed_rhs = processed_rhs[1 : len(processed_rhs)]
        output = f"{lhs}~{processed_rhs}"

        self.formula = output

        #   Remove duplicates from recode_factors
        return (
            output,
            _columns_original_order(
                columns_unordered=list(set(recoded_factors)),
                columns_ordered=recoded_factors,
            ),
        )

    def _is_factor(clause: str):
        return clause.startswith("C(") and "[T." in clause

    def _is_scale(clause: str):
        return clause.startswith("scale(")

    def formula_with_varnames_in_brackets(
        self=None,
        clause: str = "",
        df: pl.LazyFrame | pl.DataFrame | None = None,
        case_insensitive: bool = False,
        append: bool = False,
    ) -> str | FormulaBuilder:
        """
        Add a clause, but replace {var*1} with the list of variables that
            match var*1 in the dataframe

        Parameters
        ----------
        df : pl.LazyFrame | pl.DataFrame | None, optional
            Data frame to use to lookup the variable names (if not in class object).
            The default is None.
        clause : str, optional
            The default is "".
        case_insensitive : bool, optional
            Case insentive search for variables in {}.  The default is False
        append : bool, optional
            Append the Clause to the existing formula (rather than overwriting it)
            Only used if called on a FormulaBuilder object (rather than with self is None).
            The default is False
        Returns
        -------
        self (if class object) or string (if uninitialized)

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

    def exclude_variables(
        self=None,
        exclude_list: list = None,
        formula: str = "",
        df: IntoFrameT | None = None,
        case_insensitive: bool = False,
    ):
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

    def match_formula_to_columns(self=None, columns: list = None, formula: str = ""):
        if columns is None:
            columns = []
        if self is not None:
            if formula == "":
                formula = self.formula

        b_constant = FormulaBuilder.has_constant(formula=formula)
        formula = FormulaBuilder.expand(formula=formula)

        #   Separate into subclauses
        sides = formula.split("~")
        if len(sides) == 2:
            lhs = sides[0]
            rhs = sides[1]
        else:
            lhs = ""
            rhs = sides[0]

        subclauses = rhs.split("+")

        #   Sort them by string length (so longer ones match first)
        subclauses.sort(key=len, reverse=True)
        matched = []
        for clausei in subclauses:
            pattern = f"^.*{re.escape(clausei)}.*$"

            remaining = []
            for coli in columns:
                if re.match(pattern, coli):
                    if clausei not in matched:
                        matched.insert(0, clausei)
                else:
                    remaining.append(coli)
            columns = remaining

            if len(columns) == 0:
                break

        if b_constant:
            constant = "1"
        else:
            constant = "0"
        output = f"{lhs}~{constant}+{'+'.join(matched)}"

        if self is not None:
            self.formula = output

        return output
