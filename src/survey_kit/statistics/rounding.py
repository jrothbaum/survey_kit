from ..serializable import Serializable


class Rounding(Serializable):
    _save_suffix = "rounding"

    def __init__(
        self,
        round_output: bool | int = True,
        cols_round: list | None = None,
        cols_n: list | None = None,
        round_all: bool = True,
    ):
        self.round_output = round_output
        self.round_all = round_all

        if cols_round is None:
            cols_round = []
        if cols_n is None:
            cols_n = []

        self.cols_round = cols_round
        self.cols_n = cols_n
        self.cols_exclude = []

        self.set_round_digits(self.round_output)

    def set_round_digits(self, round_output: bool | int = True):
        if round_output:
            if type(round_output) is bool:
                self.round_digits = 4
            else:
                self.round_digits = self.round_output
        else:
            self.round_digits = None
