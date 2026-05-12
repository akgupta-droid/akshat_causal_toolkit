import pandas as pd
from typing import Tuple
from math import sqrt
from statistics import NormalDist


def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    treated = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    n1 = len(treated)
    n0 = len(control)

    mean1 = treated.mean()
    mean0 = control.mean()

    ate_estimate = mean1 - mean0

    var1 = treated.var(ddof=1)
    var0 = control.var(ddof=1)

    se = sqrt(var1 / n1 + var0 / n0)

    z = NormalDist().inv_cdf(1 - alpha / 2)

    ci_lower = ate_estimate - z * se
    ci_upper = ate_estimate + z * se

    return ate_estimate, ci_lower, ci_upper


def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    treated = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    n1 = len(treated)
    n0 = len(control)

    mean1 = treated.mean()
    mean0 = control.mean()

    ate_estimate = mean1 - mean0

    var1 = treated.var(ddof=1)
    var0 = control.var(ddof=1)

    se = sqrt(var1 / n1 + var0 / n0)

    t_statistic = ate_estimate / se

    normal = NormalDist()
    p_value = 2 * (1 - normal.cdf(abs(t_statistic)))

    return ate_estimate, t_statistic, p_value