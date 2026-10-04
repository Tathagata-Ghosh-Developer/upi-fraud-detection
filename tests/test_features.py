"""Check the sweep-line window features against a brute-force definition."""

import numpy as np
import pandas as pd

from src.features import add_window_features


def brute_force(df: pd.DataFrame, key: str, w: int) -> tuple[np.ndarray, np.ndarray]:
    counts, amounts = [], []
    for _, row in df.iterrows():
        past = df[(df[key] == row[key]) & (df["step"] >= row["step"] - w) & (df["step"] < row["step"])]
        counts.append(len(past))
        amounts.append(past["amount"].sum())
    return np.array(counts), np.array(amounts)


def test_window_features_match_brute_force():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "acct": rng.choice(["A", "B", "C"], size=300),
            "step": rng.integers(1, 60, size=300),
            "amount": rng.integers(1, 1000, size=300).astype(float),
            "row": np.arange(300),
        }
    )
    out = add_window_features(df, key="acct", prefix="orig", windows=[1, 24]).sort_values("row")
    for w in (1, 24):
        exp_cnt, exp_amt = brute_force(df, "acct", w)
        np.testing.assert_array_equal(out[f"orig_cnt_{w}h"].to_numpy(), exp_cnt)
        np.testing.assert_allclose(out[f"orig_amt_{w}h"].to_numpy(), exp_amt, rtol=1e-6)


def test_no_look_ahead_within_same_hour():
    # Two transactions in the same hour must not see each other.
    df = pd.DataFrame({"acct": ["A", "A", "A"], "step": [5, 5, 6], "amount": [10.0, 20.0, 5.0]})
    out = add_window_features(df, key="acct", prefix="orig", windows=[1])
    by_step = out.groupby("step")["orig_cnt_1h"].max()
    assert by_step[5] == 0
    assert by_step[6] == 2
