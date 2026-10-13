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
