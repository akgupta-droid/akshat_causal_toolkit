import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


def _parse_terms(formula: str):
    terms = [term.strip() for term in formula.split("+") if term.strip()]
    parsed = []

    for term in terms:
        if term.startswith("C(") and term.endswith(")"):
            col_name = term[2:-1].strip()   
            parsed.append((col_name, True))
        else:
            parsed.append((term, False))

    return parsed


def _build_matrix(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    parsed_terms = _parse_terms(formula)
    X_parts = []

    for col_name, force_categorical in parsed_terms:
        col = df[col_name]

        if force_categorical or not pd.api.types.is_numeric_dtype(col):
            dummies = pd.get_dummies(col, drop_first=True)
            X_parts.append(dummies.astype(float))
        else:
            X_parts.append(col.astype(float).to_frame())

    if not X_parts:
        return pd.DataFrame(index=df.index)

    return pd.concat(X_parts, axis=1)


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    X = _build_matrix(df, ps_formula)
    t = df[T].to_numpy()
    y = df[Y].to_numpy()

    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)

    ate = np.mean(((t - ps) / (ps * (1 - ps))) * y)
    return float(ate)


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    X = _build_matrix(df, formula)
    t = df[T].to_numpy()
    y = df[Y].to_numpy()

    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)

    mu1_model = LinearRegression()
    mu0_model = LinearRegression()

    mu1_model.fit(X.loc[t == 1], y[t == 1])
    mu0_model.fit(X.loc[t == 0], y[t == 0])

    mu1 = mu1_model.predict(X)
    mu0 = mu0_model.predict(X)

    treated_part = t * (y - mu1) / ps + mu1
    control_part = (1 - t) * (y - mu0) / (1 - ps) + mu0

    ate = np.mean(treated_part) - np.mean(control_part)
    return float(ate)