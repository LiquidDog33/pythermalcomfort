"""Regression checks through public PMV models after sharing their classifications."""

import numpy as np
import pytest
from scipy.optimize import brentq

from pythermalcomfort.models import pmv_ppd_ashrae, pmv_ppd_iso


@pytest.mark.parametrize("model", [pmv_ppd_iso, pmv_ppd_ashrae])
@pytest.mark.parametrize(
    ("boundary", "below", "above", "ppd"),
    [
        (-2.5, "Cold", "Cool", [93.5, 93.4]),
        (-1.5, "Cool", "Slightly Cool", [51.0, 50.8]),
        (-0.5, "Slightly Cool", "Neutral", [10.2, 10.2]),
        (0.5, "Neutral", "Slightly Warm", [10.2, 10.2]),
        (1.5, "Slightly Warm", "Warm", [50.8, 51.0]),
        (2.5, "Warm", "Hot", [93.4, 93.5]),
    ],
)
def test_pmv_models_preserve_boundary_inclusion_and_round_before_classification(
    model, boundary, below, above, ppd
):
    def calculate(temperature, *, round_output):
        return model(
            tdb=temperature,
            tr=temperature,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            limit_inputs=False,
            round_output=round_output,
        )

    # Invert the public calculation to supply real environments on either side.
    temperatures = [
        brentq(
            lambda t, target=target: calculate(t, round_output=False).pmv - target,
            -30,
            80,
        )
        for target in [boundary - 0.001, boundary + 0.001]
    ]
    unrounded = calculate(temperatures, round_output=False)
    np.testing.assert_allclose(
        unrounded.pmv, [boundary - 0.001, boundary + 0.001], atol=1e-10
    )
    np.testing.assert_array_equal(unrounded.tsv, [below, above])
    rounded = calculate(temperatures, round_output=True)
    np.testing.assert_array_equal(rounded.pmv, [boundary, boundary])
    np.testing.assert_array_equal(rounded.ppd, ppd)
    expected = above if model is pmv_ppd_iso else below
    np.testing.assert_array_equal(rounded.tsv, [expected, expected])
    scalar = calculate(temperatures[0], round_output=True)
    assert scalar.pmv == boundary
    assert scalar.tsv == expected
