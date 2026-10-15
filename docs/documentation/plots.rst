Plots
=====

This section provides an overview of the plotting capabilities available in pythermalcomfort.
The library offers class-based utilities to visualize model results with Matplotlib.
We also provide examples of how to create visually compelling and informative plots using popular Python libraries such as Matplotlib and Seaborn.

To use the plotting functions we included in pythermalcomfort, please install the optional dependencies using:

.. code-block:: bash

    pip install pythermalcomfort[plots]

Typical Use Cases
-----------------

- Comparing the results of different thermal comfort models.
- Visualizing the impact of various parameters (e.g., temperature, humidity, clothing insulation) on model output.
- Creating custom plots to communicate findings effectively.
- Analyse the comfort levels in different environments.
- Process, analyze, and visualise:
    - climate data in different formats (e.g., CSV, Excel, JSON, EPW)
    - results from building simulation software.
    - results from sensor data.

Plotting Examples
-----------------

The `pythermalcomfort.plots.matplotlib` module includes configurable plotting classes.
These classes can be used to compare model results and to visualize how variables affect outcomes.

The `plot` module has a sub-module for different backends, currently supporting Matplotlib only, but we plan to add support for other libraries in the future (e.g., Plotly, Bokeh, Seaborn).

All plotting APIs return Matplotlib handles that can be further customized.
This allows users to create richer visualizations by adding overlays or modifying existing artists.
The full list of customizable parameters is available in the class docstrings.
Use the search functionality in the top-right corner to find specific functions.

Click on the links below to show some examples of the plotting functions available in pythermalcomfort.
The full list of functions is available in the "Plotting Functions Reference" section at the end of this page.

.. toctree::

    plots/matplotlib/threshold_plot.ipynb
    plots/matplotlib/summary_plot.ipynb
    plots/matplotlib/psychrometric_plot.ipynb
    plots/matplotlib/adaptive_plot.ipynb

Practical Recipes
-----------------

These examples combine different chart types with standard Matplotlib workflows
to produce time series, overlays, and multi-panel figures.

.. toctree::

    plots/matplotlib/example_plots.ipynb

Threshold Plot
--------------

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlot
    :members: __init__, set_x_axis, set_y_axis, set_params, set_regions, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlotResult
    :members:

Summary Plot
------------

PMV region presets
~~~~~~~~~~~~~~~~~~

Use ``set_regions(preset="pmv_iso")`` or ``set_regions(preset="pmv_ashrae")``
to summarize the ``pmv`` column without entering thresholds or labels. Both use
seven thermal sensation categories: Cold, Cool, Slightly Cool, Neutral, Slightly
Warm, Warm, and Hot. Sensation is distinct from comfort-standard compliance.

The boundaries are -2.5, -1.5, -0.5, 0.5, 1.5, and 2.5. ISO includes the left
edge; ASHRAE includes the right edge. For example, PMV 0.5 is Slightly Warm in
ISO and Neutral in ASHRAE. ISO rejects values >= 10 and ASHRAE rejects values
> 10: these are output classification limits, not environmental applicability
checks or extra regions. Supplied values are classified without added rounding.

Names ignore case and surrounding whitespace. Override the default column with
``output``, and presentation with ``labels`` and ``colors`` (seven entries each;
``labels=[]`` suppresses labels). The existing seven-region palette is the default.
Units default to SI; ``units="SI"`` and ``units="IP"`` are case insensitive and
leave dimensionless PMV unchanged. Units are accepted only with a preset.
Custom ``output``/``thresholds`` calls retain left-closed boundaries and do not
convert units. A preset cannot be combined with explicit thresholds.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from pythermalcomfort.plots.matplotlib import SummaryPlot

    df = pd.DataFrame({"sensation": [-1.0, 0.0, 0.5, 1.0, np.nan]})
    # Filter explicitly; percentages describe the retained valid samples only.
    df = df.loc[np.isfinite(df["sensation"]) & (df["sensation"] < 10)]
    summary = SummaryPlot(df).set_regions(
        preset="pmv_iso",
        output="sensation",
        labels=["Cold", "Cool", "Slightly Cool", "Neutral",
                "Slightly Warm", "Warm", "Hot"],
        colors=["#527F98", "#86AEC8", "#C7DCE6", "#F1F3F2",
                "#EBCBC3", "#D88B7B", "#B85F55"],
    ).plot(title="ISO PMV thermal sensation")

Nonnumeric, missing, nonfinite, and unmapped samples raise errors; rows are never
automatically dropped. Clean nonnumeric data before filtering, for example with
``pd.to_numeric(df["sensation"], errors="coerce")``. A missing output column also
raises an error without searching for another numeric column.

When original model category assignments must be retained, use
``set_categories(result.tsv, labels=..., colors=...)`` with the same retained
rows. A model's rounding and classification order can affect assignments, so a
numeric summary cannot generally recover categories from rounded output.
Calling ``set_regions`` or ``set_categories`` replaces the preceding configuration.
A runnable model example is included as ``examples/plot_summary_pmv_presets.py``.

UTCI region preset
~~~~~~~~~~~~~~~~~~

Use ``SummaryPlot(df).set_regions(preset="utci").plot()`` to summarize the
``utci`` column in ten ordered categories, from extreme cold stress through
no thermal stress to extreme heat stress. The Celsius boundaries are -40, -27,
-13, 0, 9, 26, 32, 38, and 46, all right-closed: exactly 26 degrees Celsius
belongs to no thermal stress. Supplied values are classified without extra rounding.

Units default to SI (Celsius). Set ``units="IP"`` for Fahrenheit data; unit names
ignore case. Only thresholds and the upper classification bound are converted:
the caller supplies data in the declared units. No units are inferred and no
input data are converted. UTCI rejects values above 1000 degrees Celsius or
1832 degrees Fahrenheit, without adding an extra region or checking environmental
input applicability. Filter invalid and unmapped samples explicitly; percentages
describe the retained valid samples.

Names ignore surrounding whitespace and case. Override ``output``, or provide ten
``labels`` and ``colors``; ``labels=[]`` suppresses labels. The existing palette
for ten regions is reused. Common preset validation and configuration replacement
rules also apply to UTCI.

.. code-block:: python

    df = pd.DataFrame({"utci": [32.0, 78.8, 89.6]})  # Fahrenheit data
    summary = SummaryPlot(df).set_regions(preset="utci", units="IP").plot()

UTCI classifies a separately rounded Celsius value. For example, unrounded UTCI
values 26.049 and 26.051 degrees Celsius both display as 78.9 degrees Fahrenheit,
but the model's rounded Celsius values are 26.0 and 26.1, giving no thermal stress
and moderate heat stress. A numeric IP preset classifies both supplied 78.9 values
as moderate heat stress. Use ``set_categories(result.stress_category, ...)`` with
matching retained rows to preserve the original assignments. See the runnable
SI/IP and category-summary example ``examples/plot_summary_utci_presets.py``.

Heat Index region presets
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SummaryPlot(df).set_regions(preset="heat_index_rothfusz").plot()`` or
``preset="heat_index_schoen"`` for the corresponding model's ``hi`` column.
Both presets use the five ordered categories no risk, caution, extreme caution,
danger, extreme danger. Celsius transitions are 27, 32, 41, 54, with right-closed
intervals: exactly 32 belongs to caution. Names ignore case and surrounding
whitespace; there is no ambiguous ``hi`` alias. The existing five-region palette
is used, and ``output``, ``labels`` and ``colors`` can be overridden.

SI is the default. Both current HI models return Celsius. For ``units="IP"``,
callers must supply Fahrenheit data, converting it explicitly if needed using
``units_converter(from_units="SI", tmp=values)[0]``. Only thresholds and the
upper classification limit are converted; input data are never converted or
inferred. The upper limit, 1000 degrees Celsius or 1832 degrees Fahrenheit,
belongs to extreme danger; higher values raise an error and add no region.

Filter nonnumeric, missing, nonfinite and unmapped samples explicitly before
plotting. After filtering, percentages describe the retained valid samples.
For example, retaining five valid observations from six samples makes each
retained observation worth 20 percent. Presets do not reconstruct a model's
environmental applicability limits from its output.

Both models classify before rounding their displayed HI to one decimal.
Unrounded HI values 31.99 and 32.01 degrees Celsius therefore have different
categories (caution and extreme caution), while both display as 32.0.
A numeric preset assigns both supplied 32.0 values to caution without extra
rounding. To retain the original assignments, use ``set_categories()`` with
``result.stress_category``. See ``examples/plot_summary_heat_index_presets.py``
for runnable examples of both models, SI/IP data, presentation overrides,
explicit filtering and exact category summaries.

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlot
    :members: __init__, set_regions, set_categories, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlotResult
    :members:

Psychrometric Plot
------------------

.. autoclass:: pythermalcomfort.plots.matplotlib.PsychrometricPlot
    :members: __init__, set_x_axis, set_y_axis, set_params, set_regions, plot
    :member-order: bysource
