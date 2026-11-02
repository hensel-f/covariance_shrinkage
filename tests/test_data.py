"""Tests for the French data parser (plumbing), on a small hand-made file."""

import numpy as np
import pandas as pd
import pytest

from covariance_shrinkage.data import (
    EQUAL_WEIGHTED,
    parse_french_daily,
    parse_french_factors_daily,
)

SAMPLE = """\
This file was created using the 202608 CRSP database.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,SMALL LoBM,ME1 BM2,BIG HiBM
19260701, -99.99,   0.50,  -1.20
19260702,   1.00,  -0.25, -999

  Average Equal Weighted Returns -- Daily
,SMALL LoBM,ME1 BM2,BIG HiBM
19260701,   2.00,   0.00,   0.10
"""


def test_parses_value_weighted_returns_into_decimals_with_nan():
    df = parse_french_daily(SAMPLE)
    assert list(df.columns) == ["SMALL LoBM", "ME1 BM2", "BIG HiBM"]
    assert list(df.index) == [pd.Timestamp("1926-07-01"), pd.Timestamp("1926-07-02")]
    expected = np.array([[np.nan, 0.005, -0.012], [0.01, -0.0025, np.nan]])
    np.testing.assert_allclose(df.to_numpy(), expected)


def test_parses_another_section():
    df = parse_french_daily(SAMPLE, EQUAL_WEIGHTED)
    assert df.shape == (1, 3)
    np.testing.assert_allclose(df.to_numpy(), [[0.02, 0.0, 0.001]])


def test_unknown_section_raises():
    with pytest.raises(ValueError):
        parse_french_daily(SAMPLE, "Average Firm Size")


FACTORS_SAMPLE = """\
This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate that, over the number of trading days
compounds to 1-month TBill rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.02

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parses_factors_into_decimals():
    df = parse_french_factors_daily(FACTORS_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert list(df.index) == [pd.Timestamp("1926-07-01"), pd.Timestamp("1926-07-02")]
    np.testing.assert_allclose(df["RF"].to_numpy(), [0.0001, 0.0002])
    np.testing.assert_allclose(df.iloc[0].to_numpy(), [0.0009, -0.0023, -0.0028, 0.0001])


def test_factors_without_header_raises():
    with pytest.raises(ValueError):
        parse_french_factors_daily(SAMPLE)
