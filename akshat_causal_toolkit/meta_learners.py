import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """Estimate CATE using the S-Learner for binary treatment."""
    train_df = train.copy()
    test_df = test.copy()

    features = X + [T]
    model = LGBMRegressor()
    model.fit(train_df[features], train_df[y])

    test_t1 = test_df[X].copy()
    test_t1[T] = 1
    test_t0 = test_df[X].copy()
    test_t0[T] = 0

    mu1 = model.predict(test_t1[features])
    mu0 = model.predict(test_t0[features])

    test_df["cate"] = mu1 - mu0
    return test_df



def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """Estimate CATE using the T-Learner for binary treatment."""
    train_df = train.copy()
    test_df = test.copy()

    treated = train_df[train_df[T] == 1]
    control = train_df[train_df[T] == 0]

    mu1_model = LGBMRegressor()
    mu0_model = LGBMRegressor()

    mu1_model.fit(treated[X], treated[y])
    mu0_model.fit(control[X], control[y])

    mu1 = mu1_model.predict(test_df[X])
    mu0 = mu0_model.predict(test_df[X])

    test_df["cate"] = mu1 - mu0
    return test_df



def x_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    """Estimate CATE using the X-Learner for binary treatment."""
    train_df = train.copy()
    test_df = test.copy()

    treated = train_df[train_df[T] == 1].copy()
    control = train_df[train_df[T] == 0].copy()

    # Stage 1: outcome models
    mu1_model = LGBMRegressor()
    mu0_model = LGBMRegressor()

    mu1_model.fit(treated[X], treated[y])
    mu0_model.fit(control[X], control[y])

    # Stage 2: pseudo treatment effects
    control["tau0_pseudo"] = mu1_model.predict(control[X]) - control[y]
    treated["tau1_pseudo"] = treated[y] - mu0_model.predict(treated[X])

    tau0_model = LGBMRegressor()
    tau1_model = LGBMRegressor()

    tau0_model.fit(control[X], control["tau0_pseudo"])
    tau1_model.fit(treated[X], treated["tau1_pseudo"])

    # Propensity score model
    prop_model = LogisticRegression(penalty=None, max_iter=1000)
    prop_model.fit(train_df[X], train_df[T])
    e = prop_model.predict_proba(test_df[X])[:, 1]

    tau0_hat = tau0_model.predict(test_df[X])
    tau1_hat = tau1_model.predict(test_df[X])

    # Use the weighting formula exactly as specified in the prompt
    test_df["cate"] = e * tau0_hat + (1 - e) * tau1_hat
    return test_df



def double_ml_cate(train, test, X, T, y) -> pd.DataFrame:
    """Estimate CATE using Double ML for continuous treatment."""
    train_df = train.copy()
    test_df = test.copy()

    n = len(train_df)
    t_hat = np.zeros(n)
    y_hat = np.zeros(n)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-fitted nuisance predictions
    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        t_model = LGBMRegressor()
        y_model = LGBMRegressor()

        t_model.fit(fold_train[X], fold_train[T])
        y_model.fit(fold_train[X], fold_train[y])

        t_hat[val_idx] = t_model.predict(fold_val[X])
        y_hat[val_idx] = y_model.predict(fold_val[X])

    t_res = train_df[T].to_numpy() - t_hat
    y_res = train_df[y].to_numpy() - y_hat

    # Avoid division by values extremely close to zero
    eps = 1e-6
    safe_t_res = np.where(np.abs(t_res) < eps, np.sign(t_res) * eps, t_res)
    safe_t_res = np.where(safe_t_res == 0, eps, safe_t_res)

    y_star = y_res / safe_t_res
    w = t_res ** 2

    cate_model = LGBMRegressor()
    cate_model.fit(train_df[X], y_star, sample_weight=w)

    test_df["cate"] = cate_model.predict(test_df[X])
    return test_df
