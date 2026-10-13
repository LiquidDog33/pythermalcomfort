"""Summarize PMV thermal sensation with presets and presentation overrides."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import SummaryPlot

result = pmv_ppd_iso(
    tdb=[20, 23, 25, 27, 29, 31],
    tr=[20, 23, 25, 27, 29, 31],
    vr=0.1,
    rh=50,
    met=1.2,
    clo=0.5,
)
# Filtering is explicit. Percentages describe only the retained valid samples.
df = pd.DataFrame({"sensation": result.pmv, "original_category": result.tsv})
df = df.loc[np.isfinite(df["sensation"]) & (df["sensation"] < 10)].copy()
labels = ["Cold", "Cool", "Slightly Cool", "Neutral", "Slightly Warm", "Warm", "Hot"]
colors = ["#527F98", "#86AEC8", "#C7DCE6", "#F1F3F2", "#EBCBC3", "#D88B7B", "#B85F55"]

summary = (
    SummaryPlot(df)
    .set_regions(
        preset="pmv_iso",
        output="sensation",
        labels=labels,
        colors=colors,
    )
    .plot(title="ISO PMV thermal sensation")
)
print(summary.percentages)
# Use the returned categories when original model assignments must be retained.
SummaryPlot(df).set_categories(
    df["original_category"],
    labels=labels,
    colors=colors,
).plot(title="Original ISO model categories")
# Neutral sensation alone does not establish comfort-standard compliance.
plt.show()
