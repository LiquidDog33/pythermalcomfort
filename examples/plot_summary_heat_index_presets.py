"""Summarize Rothfusz and Schoen HI values or their original categories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import heat_index_rothfusz, heat_index_schoen
from pythermalcomfort.plots.matplotlib import SummaryPlot
from pythermalcomfort.utilities import units_converter

labels = ["no risk", "caution", "extreme caution", "danger", "extreme danger"]
colors = ["green", "yellow", "orange", "red", "black"]
for preset, model in [
    ("heat_index_rothfusz", heat_index_rothfusz),
    ("heat_index_schoen", heat_index_schoen),
]:
    # Both models return Celsius. NaN represents an invalid or missing sample;
    # Rothfusz also returns NaN for dry-bulb temperatures below 27 C by default.
    result = model(tdb=[28, 30, 35, 40, 45, np.nan], rh=50)
    df = pd.DataFrame({"hi": result.hi, "original_category": result.stress_category})
    # Explicitly retain finite mapped outputs. Every percentage describes these
    # five retained observations, rather than all six original samples.
    df = df.loc[np.isfinite(df["hi"]) & (df["hi"] <= 1000)].copy()
    df["fahrenheit"] = units_converter(from_units="SI", tmp=df["hi"].to_numpy())[0]
    si = SummaryPlot(df).set_regions(preset=preset).plot(title=f"{preset}: Celsius")
    ip = (
        SummaryPlot(df)
        .set_regions(
            preset=preset,
            units="IP",
            output="fahrenheit",
            labels=["Low", "Caution", "Elevated", "High", "Severe"],
            colors=colors,
        )
        .plot(title=f"{preset}: caller-converted Fahrenheit")
    )
    print(si.percentages)
    print(ip.percentages)
    # Models classify before display rounding. HI 31.99 and 32.01 C both
    # display as 32.0 C, yet retain caution and extreme caution, respectively.
    # Numeric presets classify supplied displays; use original categories when
    # those exact model assignments are needed.
    SummaryPlot(df).set_categories(
        df["original_category"],
        labels=labels,
        colors=colors,
    ).plot(title=f"{preset}: original categories")
plt.show()
