# Akshat Causal Toolkit

[![Tests](https://github.com/yourusername/akshat-causal-toolkit/workflows/Tests/badge.svg)](https://github.com/
akgupta-droid/akshat-causal-toolkit/actions)
[![PyPI version](https://badge.fury.io/py/akshat-causal-toolkit.svg)](https://pypi.org/project/akshat-causal-toolkit/)

A Python package for causal inference methods including ATE estimation, propensity scores, and meta-learners.

## Installation

```bash
pip install akshat-causal-toolkit
```

## Quick Start

```python
from akshat_causal_toolkit import (
    calculate_ate_ci,
    calculate_ate_pvalue,
    ipw,
    doubly_robust,
    s_learner_discrete,
    t_learner_discrete,
    x_learner_discrete,
    double_ml_cate,
)
```

