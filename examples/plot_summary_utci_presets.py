"""Summarize UTCI in Celsius, Fahrenheit, or original model categories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import utci
from pythermalcomfort.plots.matplotlib import SummaryPlot
from pythermalcomfort.utilities import units_converter

result = utci(
    tdb=[-10, 0, 15, 25, 35, 45, 60], tr=[-10, 0, 15, 25, 35, 45, 60], v=1, rh=50
)
df = pd.DataFrame({"utci": result.utci, "original_category": result.stress_category})
# Filtering is explicit. Percentages describe only the retained valid samples.
df = df.loc[np.isfinite(df["utci"]) & (df["utci"] <= 1000)].copy()
df["fahrenheit"] = units_converter(from_units="SI", tmp=df["utci"].to_numpy())[0]
labels = [
    "extreme cold stress",
    "very strong cold stress",
    "strong cold stress",
    "moderate cold stress",
    "slight cold stress",
    "no thermal stress",
    "moderate heat stress",
    "strong heat stress",
    "very strong heat stress",
    "extreme heat stress",
]
colors = [
    "#082567",
    "#174A8B",
    "#286DB2",
    "#5B9BCB",
    "#A5CEE3",
    "#EEEEEE",
    "#FDD49E",
    "#FC8D59",
    "#E34A33",
    "#990000",
]
si = SummaryPlot(df).set_regions(preset="utci").plot(title="UTCI in Celsius")
ip = (
    SummaryPlot(df)
    .set_regions(
        preset="utci",
        units="IP",
        output="fahrenheit",
        labels=labels,
        colors=colors,
    )
    .plot(title="UTCI in Fahrenheit (caller-converted data)")
)
print(si.percentages)
print(ip.percentages)
# Rounded IP model displays cannot always recover the model's SI categories:
# 26.049 / 26.051 C both display as 78.9 F, but classify from 26.0 / 26.1 C.
SummaryPlot(df).set_categories(
    df["original_category"],
    labels=labels,
    colors=colors,
).plot(title="Original UTCI model categories")
plt.show()
