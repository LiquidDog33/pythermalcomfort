from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from pythermalcomfort.plots.matplotlib._shared import _default_region_colors
from pythermalcomfort.plots.matplotlib.summary import (
    SummaryPlot,
    SummaryPlotResult,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


@pytest.fixture
def pmv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tdb": [20.0, 25.0, 30.0],
            "rh": [50.0, 50.0, 50.0],
            "pmv": [-0.6, 0.0, 0.7],
        }
    )


def _new_summary(pmv_df: pd.DataFrame) -> SummaryPlot:
    return SummaryPlot(pmv_df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


@pytest.mark.parametrize(
    ("preset", "expected"),
    [("pmv_iso", [0, 0, 0, 0, 100, 0, 0]), ("pmv_ashrae", [0, 0, 0, 100, 0, 0, 0])],
)
def test_pmv_presets_distinguish_thermal_sensation_at_boundary(preset, expected):
    result = SummaryPlot(pd.DataFrame({"pmv": [0.5]})).set_regions(preset=preset).plot()
    assert result.percentages.index.tolist() == [
        "Cold",
        "Cool",
        "Slightly Cool",
        "Neutral",
        "Slightly Warm",
        "Warm",
        "Hot",
    ]
    assert result.percentages.tolist() == expected
    assert isinstance(result, SummaryPlotResult)


@pytest.mark.parametrize(
    ("preset", "value"),
    [("pmv_iso", 10.0), ("pmv_iso", 10.001), ("pmv_ashrae", 10.001)],
)
def test_pmv_presets_reject_unmapped_samples(preset, value):
    with pytest.raises(ValueError, match="Filter.*percentages"):
        SummaryPlot(pd.DataFrame({"pmv": [value]})).set_regions(preset=preset)


@pytest.mark.parametrize("units", ["SI", "si", "IP", "ip"])
def test_pmv_preset_units_do_not_change_dimensionless_values(units):
    result = (
        SummaryPlot(pd.DataFrame({"pmv": [0.5]}))
        .set_regions(preset=" PMV_ASHRAE ", units=units)
        .plot()
    )
    assert result.percentages["Neutral"] == 100


def test_region_units_are_only_valid_in_preset_mode(pmv_df):
    with pytest.raises(ValueError, match="SI.*IP"):
        SummaryPlot(pmv_df).set_regions(preset="pmv_iso", units="kelvin")
    with pytest.raises(ValueError, match="units.*preset"):
        SummaryPlot(pmv_df).set_regions(output="pmv", thresholds=[0], units="IP")


@pytest.mark.parametrize(
    ("n_regions", "expected"),
    [
        (2, ["#86AEC8", "#D88B7B"]),
        (3, ["#86AEC8", "#F1F3F2", "#D88B7B"]),
        (5, ["#5F8FA9", "#B7D0DE", "#F1F3F2", "#E7B7AC", "#C66B5E"]),
        (
            7,
            [
                "#527F98",
                "#86AEC8",
                "#C7DCE6",
                "#F1F3F2",
                "#EBCBC3",
                "#D88B7B",
                "#B85F55",
            ],
        ),
    ],
)
def test_default_region_palette_progresses_from_cool_to_warm(
    n_regions: int, expected: list[str]
) -> None:
    assert _default_region_colors(n_regions) == expected


def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SummaryPlot([1, 2, 3])


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        SummaryPlot(pd.DataFrame())


def test_plot_requires_set_regions(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Call set_regions"):
        SummaryPlot(pmv_df).plot()


def test_set_regions_rejects_empty_output_name(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        SummaryPlot(pmv_df).set_regions(output="   ", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_missing_output_column(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="output column 'utci' was not found in the DataFrame.",
    ):
        SummaryPlot(pmv_df).set_regions(output="utci", thresholds=[9, 26])


def test_set_regions_rejects_wrong_label_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        SummaryPlot(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cold", "Hot"],
        )


def test_set_regions_rejects_wrong_color_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        SummaryPlot(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            colors=["#4c78a8", "#e15759"],
        )


def test_plot_uses_provided_axis(pmv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()

    result = _new_summary(pmv_df).plot(ax=ax)

    assert result.ax is ax
    assert result.fig is fig


def test_plot_vertical_mode_executes(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True)

    assert isinstance(result, SummaryPlotResult)
    assert len(result.artists) > 0


def test_plot_uses_compact_default_figsize(pmv_df: pd.DataFrame) -> None:
    horizontal = _new_summary(pmv_df).plot()
    vertical = _new_summary(pmv_df).plot(vertical=True)

    assert tuple(horizontal.fig.get_size_inches()) == pytest.approx((6.4, 1.8))
    assert tuple(vertical.fig.get_size_inches()) == pytest.approx((2.8, 4.0))


def test_vertical_empty_labels_uses_compact_xlim(pmv_df: pd.DataFrame) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot(vertical=True, legend=False)
    )

    left, right = result.ax.get_xlim()
    assert right - left == pytest.approx(1.0)


def test_plot_bar_kws_apply_to_horizontal_bars(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(bar_kws={"alpha": 0.4, "linewidth": 2.5})
    patch = result.artists[0].patches[0]

    assert patch.get_alpha() == pytest.approx(0.4)
    assert patch.get_linewidth() == pytest.approx(2.5)


def test_plot_bar_kws_apply_to_vertical_bars(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True, bar_kws={"hatch": "//"})
    patch = result.artists[0].patches[0]

    assert patch.get_hatch() == "//"


def test_plot_bar_kws_can_override_horizontal_bar_height(
    pmv_df: pd.DataFrame,
) -> None:
    result = _new_summary(pmv_df).plot(bar_kws={"height": 0.42})
    patch = result.artists[0].patches[0]

    assert patch.get_height() == pytest.approx(0.42)


def test_plot_bar_kws_can_override_vertical_bar_width(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True, bar_kws={"width": 0.52})
    patch = result.artists[0].patches[0]

    assert patch.get_width() == pytest.approx(0.52)


def test_plot_returns_percentages(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    expected_labels = ["PMV < -0.5", "-0.5 ≤ PMV < 0.5", "PMV ≥ 0.5"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_plot_uses_custom_labels_when_provided(pmv_df: pd.DataFrame) -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = (
        SummaryPlot(pmv_df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=custom_labels,
        )
        .plot()
    )

    assert result.percentages.index.tolist() == custom_labels


def test_plot_supports_utci_like_existing_column() -> None:
    df = pd.DataFrame(
        {
            "tdb": [10.0, 24.0, 32.0],
            "utci": [5.0, 18.0, 28.0],
        }
    )

    result = SummaryPlot(df).set_regions(output="utci", thresholds=[9, 26]).plot()

    expected_labels = ["UTCI < 9", "9 ≤ UTCI < 26", "UTCI ≥ 26"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_set_regions_rejects_non_numeric_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, "bad", 0.2]})

    with pytest.raises(ValueError, match="non-numeric"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_non_finite_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, float("inf"), 0.2]})

    with pytest.raises(ValueError, match="non-finite"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_summary_with_custom_labels() -> None:
    df = pd.DataFrame({"pmv": [0.7, -0.3, 0.1, -0.8, 1.2]})
    result = (
        SummaryPlot(df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cool", "Comfortable", "Warm"],
        )
        .plot()
    )
    assert isinstance(result, SummaryPlotResult)
    assert list(result.percentages.index) == ["Cool", "Comfortable", "Warm"]


def test_summary_handles_numeric_string_column() -> None:
    df = pd.DataFrame({"pmv": ["0.7", "-0.3", "0.1", "-0.8", "1.2"]})
    result = SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5]).plot()
    assert isinstance(result, SummaryPlotResult)
    assert result.percentages.sum() > 99.9


def test_plot_legend_shown_by_default(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert isinstance(result.legend, Legend)


def test_plot_title_does_not_overlap_horizontal_legend(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(title="PMV Distribution")
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()

    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)

    assert title_bbox.y0 > legend_bbox.y1


def test_plot_title_does_not_overlap_vertical_legend(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(
        vertical=True,
        title="PMV Distribution (Vertical)",
    )
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()

    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)

    assert title_bbox.y0 > legend_bbox.y1


def test_plot_legend_none_when_disabled(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(legend=False)

    assert result.legend is None


def test_plot_result_has_no_data_attribute(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert not hasattr(result, "data")


def test_set_regions_empty_labels_suppresses_label_text(pmv_df: pd.DataFrame) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


def test_empty_labels_suppresses_label_text_via_thresholds(
    pmv_df: pd.DataFrame,
) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


# ── set_categories ──────────────────────────────────────────────────────────


@pytest.fixture
def categories_df() -> pd.DataFrame:
    return pd.DataFrame({"row": range(10)})


def test_set_categories_returns_expected_percentages(
    categories_df: pd.DataFrame,
) -> None:
    categories = ["A"] * 3 + ["B"] * 7
    result = (
        SummaryPlot(categories_df)
        .set_categories(
            categories, labels=["A", "B", "C"], colors=["#111", "#222", "#333"]
        )
        .plot()
    )

    assert result.percentages.tolist() == [30.0, 70.0, 0.0]
    assert list(result.percentages.index) == ["A", "B", "C"]


def test_set_categories_zero_count_category_renders_at_zero(
    categories_df: pd.DataFrame,
) -> None:
    categories = ["A"] * 10
    result = (
        SummaryPlot(categories_df)
        .set_categories(categories, labels=["A", "B"], colors=["#111", "#222"])
        .plot(legend=False)
    )

    assert result.percentages["B"] == 0.0


def test_set_categories_rejects_length_mismatch(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="one value per row"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 9, labels=["A"], colors=["#111"]
        )


def test_set_categories_rejects_unknown_value(categories_df: pd.DataFrame) -> None:
    categories = ["A"] * 9 + ["Z"]
    with pytest.raises(ValueError, match="not present in labels"):
        SummaryPlot(categories_df).set_categories(
            categories, labels=["A"], colors=["#111"]
        )


def test_set_categories_rejects_labels_colors_length_mismatch(
    categories_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="same length"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A", "B"], colors=["#111"]
        )


def test_set_categories_rejects_duplicate_labels(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A", "A"], colors=["#111", "#222"]
        )


def test_set_categories_rejects_invalid_color(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid color"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A"], colors=["not-a-color"]
        )


def test_set_categories_clears_region_config(pmv_df: pd.DataFrame) -> None:
    sp = _new_summary(pmv_df)
    assert sp._region_config is not None

    sp.set_categories(["A", "B", "A"], labels=["A", "B"], colors=["#111", "#222"])
    assert sp._region_config is None
    assert sp._categories is not None


def test_set_regions_clears_categories(pmv_df: pd.DataFrame) -> None:
    sp = SummaryPlot(pmv_df).set_categories(
        ["A", "B", "A"], labels=["A", "B"], colors=["#111", "#222"]
    )
    assert sp._categories is not None

    sp.set_regions(output="pmv", thresholds=[-0.5, 0.5])
    assert sp._categories is None
    assert sp._category_labels is None
    assert sp._category_colors is None


def test_set_categories_adaptive_np_select_recipe_end_to_end() -> None:
    """Reproduces the np.select recipe documented on set_categories()."""
    np = pytest.importorskip("numpy")
    from pythermalcomfort.models import adaptive_ashrae

    df = pd.DataFrame(
        {
            "tdb": [22.0, 25.0, 30.0, 19.0],
            "tr": [22.0, 25.0, 30.0, 19.0],
            "t_rm": [20.0, 20.0, 20.0, 20.0],
            "v": [0.1, 0.1, 0.1, 0.1],
        }
    )
    result = adaptive_ashrae(
        tdb=df["tdb"], tr=df["tr"], t_running_mean=df["t_rm"], v=df["v"]
    )
    categories = np.select(
        [result.acceptability_90, result.acceptability_80],
        ["90% Acceptability", "80% Acceptability"],
        default="Outside",
    )

    plot_result = (
        SummaryPlot(df)
        .set_categories(
            categories,
            labels=["90% Acceptability", "80% Acceptability", "Outside"],
            colors=["#6BB3FF", "#B3D9FF", "#D9D9D9"],
        )
        .plot(title="Adaptive comfort distribution")
    )

    # Rows 0-1 (tdb=tr=22/25) satisfy acceptability_90; rows 2-3 (tdb=tr=30/19)
    # satisfy neither band at t_rm=20.
    assert plot_result.percentages.to_dict() == {
        "90% Acceptability": 50.0,
        "80% Acceptability": 0.0,
        "Outside": 50.0,
    }


@pytest.mark.parametrize("preset", ["pmv_iso", "pmv_ashrae"])
@pytest.mark.parametrize(
    ("boundary", "lower", "upper"),
    [
        (-2.5, "Cold", "Cool"),
        (-1.5, "Cool", "Slightly Cool"),
        (-0.5, "Slightly Cool", "Neutral"),
        (0.5, "Neutral", "Slightly Warm"),
        (1.5, "Slightly Warm", "Warm"),
        (2.5, "Warm", "Hot"),
    ],
)
def test_pmv_presets_classify_each_boundary_without_rounding(
    preset, boundary, lower, upper
):
    # A tiny offset must remain on its own side, even if PMV rounds to the boundary.
    result = (
        SummaryPlot(pd.DataFrame({"pmv": [boundary - 1e-8, boundary, boundary + 1e-8]}))
        .set_regions(preset=preset)
        .plot()
    )
    expected_lower = 33.3 if preset == "pmv_iso" else 66.7
    assert result.percentages[lower] == expected_lower
    assert result.percentages[upper] == 100 - expected_lower
    assert (result.percentages == 0).sum() == 5


@pytest.mark.parametrize(
    ("preset", "value"),
    [("pmv_iso", 9.999999), ("pmv_ashrae", 10.0), ("pmv_iso", -100)],
)
def test_pmv_presets_accept_mapped_extreme_values(preset, value):
    result = (
        SummaryPlot(pd.DataFrame({"pmv": [value]})).set_regions(preset=preset).plot()
    )
    assert result.percentages["Cold" if value < 0 else "Hot"] == 100
    assert len(result.ax.patches) == 7


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), "bad", None]
)
def test_pmv_presets_reject_invalid_samples_with_filtering_guidance(value):
    with pytest.raises(ValueError, match="Filter.*percentages"):
        SummaryPlot(pd.DataFrame({"pmv": [0, value]})).set_regions(preset="pmv_iso")


@pytest.mark.parametrize("preset", ["pmv", "pmv_iss", "utc", ""])
def test_unknown_presets_list_supported_identifiers(pmv_df, preset):
    with pytest.raises(ValueError, match="Supported presets: pmv_iso, pmv_ashrae"):
        SummaryPlot(pmv_df).set_regions(preset=preset)


def test_presets_validate_configuration_and_default_column(pmv_df):
    with pytest.raises(ValueError, match="explicit thresholds"):
        SummaryPlot(pmv_df).set_regions(preset="pmv_iso", thresholds=[0])
    with pytest.raises(ValueError, match="'pmv'.*not found"):
        SummaryPlot(pd.DataFrame({"other": [0]})).set_regions(preset="pmv_iso")
    with pytest.raises(TypeError, match="preset must be a string"):
        SummaryPlot(pmv_df).set_regions(preset=1)
    with pytest.raises(TypeError, match="output and thresholds are required"):
        SummaryPlot(pmv_df).set_regions(output="pmv")
    with pytest.raises(TypeError, match="output and thresholds are required"):
        SummaryPlot(pmv_df).set_regions(thresholds=[0])
    for overrides, message in [
        ({"labels": ["one"]}, "labels must have length 7"),
        ({"colors": ["red"]}, "colors must have length 7"),
        ({"colors": ["invalid"] * 7}, "Invalid color"),
        ({"output": 5}, "output must be a string"),
        ({"output": "missing"}, "not found"),
        ({"units": 1}, "SI.*IP"),
    ]:
        with pytest.raises((TypeError, ValueError), match=message):
            SummaryPlot(pmv_df).set_regions(preset="pmv_iso", **overrides)


def test_preset_presentation_overrides_do_not_change_defaults():
    from matplotlib.colors import to_rgba

    df = pd.DataFrame({"other": [0.5]})
    labels = ["one", "two", "three", "four", "five", "six", "seven"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "black"]
    builder = SummaryPlot(df)
    assert (
        builder.set_regions(
            preset="pmv_iso", output="other", labels=labels, colors=colors
        )
        is builder
    )
    result = builder.plot()
    assert result.percentages.index.tolist() == labels
    assert result.percentages.tolist() == [0, 0, 0, 0, 100, 0, 0]
    assert [p.get_facecolor() for p in result.ax.patches] == [
        to_rgba(c) for c in colors
    ]
    assert [t.get_text() for t in result.legend.get_texts()] == labels
    labels[4] = "mutated"
    colors[4] = "white"
    assert builder.plot().percentages.index[4] == "five"
    default = builder.set_regions(preset="pmv_iso", output="other").plot()
    assert default.percentages.index[4] == "Slightly Warm"
    assert [p.get_facecolor() for p in default.ax.patches] == [
        to_rgba(c)
        for c in [
            "#527F98",
            "#86AEC8",
            "#C7DCE6",
            "#F1F3F2",
            "#EBCBC3",
            "#D88B7B",
            "#B85F55",
        ]
    ]
    empty_labels = builder.set_regions(
        preset="pmv_iso", output="other", labels=[]
    ).plot(legend=False)
    assert empty_labels.percentages.index.tolist() == [""] * 7


def test_presets_custom_thresholds_and_categories_replace_each_other():
    builder = SummaryPlot(pd.DataFrame({"pmv": [0.5]}))
    assert builder.set_regions(preset="pmv_ashrae").plot().percentages["Neutral"] == 100
    assert (
        builder.set_regions(preset="pmv_iso").plot().percentages["Slightly Warm"] == 100
    )
    custom = builder.set_regions(
        output="pmv", thresholds=[0.5], labels=["below", "above"]
    ).plot()
    assert custom.percentages.tolist() == [0, 100]
    assert (
        builder.set_categories(["original"], labels=["original"], colors=["red"])
        is builder
    )
    assert builder.plot().percentages.to_dict() == {"original": 100}
    assert builder.set_regions(preset="pmv_ashrae").plot().percentages["Neutral"] == 100
    builder.set_categories(["original"], labels=["original"], colors=["red"])
    assert builder.set_regions(
        output="pmv", thresholds=[0]
    ).plot().percentages.tolist() == [0, 100]


def test_preset_validates_samples_again_when_dataframe_is_changed():
    df = pd.DataFrame({"pmv": [0]})
    builder = SummaryPlot(df).set_regions(preset="pmv_iso")
    df.loc[0, "pmv"] = 10
    with pytest.raises(ValueError, match="unmapped"):
        builder.plot()


@pytest.mark.parametrize(
    ("boundary", "lower", "upper"),
    [
        (-40, "extreme cold stress", "very strong cold stress"),
        (-27, "very strong cold stress", "strong cold stress"),
        (-13, "strong cold stress", "moderate cold stress"),
        (0, "moderate cold stress", "slight cold stress"),
        (9, "slight cold stress", "no thermal stress"),
        (26, "no thermal stress", "moderate heat stress"),
        (32, "moderate heat stress", "strong heat stress"),
        (38, "strong heat stress", "very strong heat stress"),
        (46, "very strong heat stress", "extreme heat stress"),
    ],
)
@pytest.mark.parametrize("units", ["SI", "ip"])
def test_utci_preset_classifies_every_boundary(boundary, lower, upper, units):
    if units == "ip":
        boundary = boundary * 9 / 5 + 32
    result = (
        SummaryPlot(
            pd.DataFrame({"utci": [boundary - 1e-8, boundary, boundary + 1e-8]})
        )
        .set_regions(preset="utci", units=units)
        .plot()
    )
    assert result.percentages[lower] == 66.7
    assert result.percentages[upper] == 33.3
    assert (result.percentages == 0).sum() == 8


@pytest.mark.parametrize(("units", "limit"), [("SI", 1000), ("IP", 1832)])
def test_utci_preset_preserves_upper_classification_limit(units, limit):
    df = pd.DataFrame({"utci": [float(limit)]})
    builder = SummaryPlot(df).set_regions(preset="utci", units=units)
    result = builder.plot()
    assert result.percentages["extreme heat stress"] == 100
    assert len(result.ax.patches) == 10
    df.loc[0, "utci"] = limit + 1e-6
    with pytest.raises(ValueError, match="unmapped.*Filter.*retained valid"):
        builder.plot()
    with pytest.raises(ValueError, match="unmapped.*Filter"):
        builder.set_regions(preset="utci", units=units)


UTCI_LABELS = [
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


def test_utci_presets_default_si_and_equivalent_ip_data():
    values = [-45, -30, -20, -5, 5, 20, 30, 35, 40, 50]
    df = pd.DataFrame({"utci": values, "fahrenheit": [v * 9 / 5 + 32 for v in values]})
    si = SummaryPlot(df).set_regions(preset="utci").plot()
    ip = (
        SummaryPlot(df)
        .set_regions(preset=" UTCI ", units="iP", output="fahrenheit")
        .plot()
    )
    assert si.percentages.index.tolist() == UTCI_LABELS
    assert si.percentages.tolist() == [10] * 10
    pd.testing.assert_series_equal(si.percentages, ip.percentages)
    assert [p.get_facecolor() for p in si.ax.patches] == [
        p.get_facecolor() for p in ip.ax.patches
    ]
    # Declaring units never transforms or guesses the supplied data.
    unchanged = (
        SummaryPlot(pd.DataFrame({"utci": [26]}))
        .set_regions(preset="utci", units="IP")
        .plot()
    )
    assert unchanged.percentages["moderate cold stress"] == 100


def test_utci_preset_presentation_and_configuration_switching():
    from matplotlib.colors import to_rgba

    df = pd.DataFrame({"utci": [26], "pmv": [0.5]})
    builder = SummaryPlot(df)
    default = builder.set_regions(preset="utci").plot()
    labels = [f"Region {i}" for i in range(10)]
    colors = [
        "navy",
        "blue",
        "cyan",
        "teal",
        "green",
        "white",
        "yellow",
        "orange",
        "red",
        "black",
    ]
    assert builder.set_regions(preset="utci", labels=labels, colors=colors) is builder
    result = builder.plot()
    assert result.percentages.index.tolist() == labels
    assert result.percentages.tolist() == [0] * 5 + [100] + [0] * 4
    assert [p.get_facecolor() for p in result.ax.patches] == [
        to_rgba(c) for c in colors
    ]
    assert [t.get_text() for t in result.legend.get_texts()] == labels
    fresh = SummaryPlot(df).set_regions(preset="utci").plot()
    assert [p.get_facecolor() for p in fresh.ax.patches] == [
        p.get_facecolor() for p in default.ax.patches
    ]
    assert (
        builder.set_regions(preset="utci", labels=[]).plot().percentages.index.tolist()
        == [""] * 10
    )
    assert (
        builder.set_regions(preset="pmv_iso").plot().percentages["Slightly Warm"] == 100
    )
    assert (
        builder.set_regions(preset="utci").plot().percentages["no thermal stress"]
        == 100
    )
    custom = builder.set_regions(
        output="utci", thresholds=[26], labels=["low", "high"]
    ).plot()
    assert custom.percentages.tolist() == [0, 100]
    builder.set_categories(["original"], labels=["original"], colors=["red"])
    assert builder.plot().percentages.to_dict() == {"original": 100}
    assert (
        builder.set_regions(preset="utci", units="IP")
        .plot()
        .percentages["moderate cold stress"]
        == 100
    )
    builder.set_categories(["original"], labels=["original"], colors=["red"])
    assert builder.plot().percentages.to_dict() == {"original": 100}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"thresholds": [0]}, "explicit thresholds"),
        ({"units": "C"}, "SI.*IP"),
        ({"units": 1}, "SI.*IP"),
        ({"output": "missing"}, "not found"),
        ({"labels": ["one"]}, "labels must have length 10"),
        ({"colors": ["red"]}, "colors must have length 10"),
        ({"colors": ["invalid"] * 10}, "Invalid color"),
    ],
)
def test_utci_preset_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        SummaryPlot(pd.DataFrame({"utci": [26]})).set_regions(
            preset="utci", **overrides
        )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), "bad", None]
)
def test_utci_preset_rejects_invalid_samples(value):
    with pytest.raises(ValueError, match="Filter.*retained valid"):
        SummaryPlot(pd.DataFrame({"utci": [26, value]})).set_regions(preset="utci")


def test_utci_preset_requires_default_column_without_numeric_fallback():
    with pytest.raises(ValueError, match="'utci'.*not found"):
        SummaryPlot(pd.DataFrame({"other": [26]})).set_regions(preset="utci")


def test_rounded_utci_numeric_summary_and_original_categories_have_distinct_contracts():
    # 26.049 and 26.051 C both display as 78.9 F, but round to 26.0 and 26.1 C
    # for the model's classification (covered through the public UTCI model tests).
    df = pd.DataFrame({"utci": [78.9, 78.9]})
    numeric = SummaryPlot(df).set_regions(preset="utci", units="IP").plot()
    assert numeric.percentages["moderate heat stress"] == 100
    original = (
        SummaryPlot(df)
        .set_categories(
            ["no thermal stress", "moderate heat stress"],
            labels=["no thermal stress", "moderate heat stress"],
            colors=["green", "orange"],
        )
        .plot()
    )
    assert original.percentages.to_dict() == {
        "no thermal stress": 50,
        "moderate heat stress": 50,
    }


@pytest.mark.parametrize("preset", ["heat_index_rothfusz", "heat_index_schoen"])
@pytest.mark.parametrize(
    ("boundary", "lower", "upper"),
    [
        (27, "no risk", "caution"),
        (32, "caution", "extreme caution"),
        (41, "extreme caution", "danger"),
        (54, "danger", "extreme danger"),
    ],
)
@pytest.mark.parametrize("units", ["sI", "ip"])
def test_heat_index_preset_classifies_every_boundary(
    preset, boundary, lower, upper, units
):
    if units == "ip":
        boundary = boundary * 9 / 5 + 32
    result = (
        SummaryPlot(pd.DataFrame({"hi": [boundary - 1e-8, boundary, boundary + 1e-8]}))
        .set_regions(preset=preset, units=units)
        .plot()
    )
    assert result.percentages[lower] == 66.7
    assert result.percentages[upper] == 33.3
    assert (result.percentages == 0).sum() == 3


HI_LABELS = ["no risk", "caution", "extreme caution", "danger", "extreme danger"]
HI_PRESETS = ["heat_index_rothfusz", "heat_index_schoen"]


@pytest.mark.parametrize("preset", HI_PRESETS)
@pytest.mark.parametrize(("units", "limit"), [("SI", 1000), ("IP", 1832)])
def test_heat_index_preset_upper_limit(preset, units, limit):
    df = pd.DataFrame({"hi": [float(limit)]})
    builder = SummaryPlot(df).set_regions(preset=preset, units=units)
    result = builder.plot()
    assert result.percentages.to_dict() == dict(
        zip(HI_LABELS, [0, 0, 0, 0, 100], strict=True)
    )
    assert len(result.ax.patches) == 5
    df.loc[0, "hi"] = limit + 1e-6
    with pytest.raises(ValueError, match="unmapped.*Filter.*retained valid"):
        builder.plot()
    with pytest.raises(ValueError, match="unmapped.*Filter"):
        builder.set_regions(preset=preset, units=units)


@pytest.mark.parametrize("preset", HI_PRESETS)
def test_heat_index_preset_units_and_presentation(preset):
    from matplotlib.colors import to_rgba

    df = pd.DataFrame(
        {"hi": [20, 30, 35, 45, 60], "fahrenheit": [68, 86, 95, 113, 140]}
    )
    si = SummaryPlot(df).set_regions(preset=preset).plot()
    ip = (
        SummaryPlot(df)
        .set_regions(preset=f" {preset.upper()} ", units="iP", output="fahrenheit")
        .plot()
    )
    assert si.percentages.index.tolist() == HI_LABELS
    assert si.percentages.tolist() == [20] * 5
    pd.testing.assert_series_equal(si.percentages, ip.percentages)
    assert [p.get_facecolor() for p in si.ax.patches] == [
        p.get_facecolor() for p in ip.ax.patches
    ]
    labels = ["Safe", "Mild", "Elevated", "High", "Severe"]
    colors = ["green", "yellow", "orange", "red", "black"]
    builder = SummaryPlot(df)
    assert builder.set_regions(preset=preset, labels=labels, colors=colors) is builder
    custom = builder.plot()
    assert custom.percentages.index.tolist() == labels
    assert custom.percentages.tolist() == [20] * 5
    assert [t.get_text() for t in custom.legend.get_texts()] == labels
    assert [p.get_facecolor() for p in custom.ax.patches] == [
        to_rgba(c) for c in colors
    ]
    fresh = SummaryPlot(df).set_regions(preset=preset).plot()
    assert [p.get_facecolor() for p in fresh.ax.patches] == [
        p.get_facecolor() for p in si.ax.patches
    ]
    assert (
        builder.set_regions(preset=preset, labels=[]).plot().percentages.index.tolist()
        == [""] * 5
    )
    # The units declaration changes thresholds only, never the supplied Celsius values.
    unchanged = SummaryPlot(df).set_regions(preset=preset, units="IP").plot()
    assert unchanged.percentages["no risk"] == 100


@pytest.mark.parametrize("preset", HI_PRESETS)
def test_heat_index_preset_switches_configuration(preset):
    builder = SummaryPlot(pd.DataFrame({"hi": [32], "pmv": [0.5]}))
    for other in [preset, "pmv_iso", *HI_PRESETS]:
        result = builder.set_regions(preset=other).plot()
        assert (
            result.percentages["Slightly Warm" if other == "pmv_iso" else "caution"]
            == 100
        )
    custom = builder.set_regions(
        output="hi", thresholds=[32], labels=["low", "high"]
    ).plot()
    assert custom.percentages.tolist() == [0, 100]
    builder.set_categories(["original"], labels=["original"], colors=["red"])
    assert builder.plot().percentages.to_dict() == {"original": 100}
    assert builder.set_regions(preset=preset).plot().percentages["caution"] == 100
    builder.set_categories(["original"], labels=["original"], colors=["red"])
    assert builder.plot().percentages.to_dict() == {"original": 100}


@pytest.mark.parametrize("preset", HI_PRESETS)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"thresholds": [0]}, "explicit thresholds"),
        ({"units": "C"}, "SI.*IP"),
        ({"units": 1}, "SI.*IP"),
        ({"output": "missing"}, "not found"),
        ({"labels": ["one"]}, "labels must have length 5"),
        ({"colors": ["red"]}, "colors must have length 5"),
        ({"colors": ["invalid"] * 5}, "Invalid color"),
    ],
)
def test_heat_index_preset_rejects_invalid_configuration(preset, overrides, message):
    with pytest.raises(ValueError, match=message):
        SummaryPlot(pd.DataFrame({"hi": [32]})).set_regions(preset=preset, **overrides)


@pytest.mark.parametrize("preset", HI_PRESETS)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), "bad", None]
)
def test_heat_index_preset_rejects_invalid_samples(preset, value):
    with pytest.raises(ValueError, match="Filter.*retained valid"):
        SummaryPlot(pd.DataFrame({"hi": [32, value]})).set_regions(preset=preset)


@pytest.mark.parametrize("preset", HI_PRESETS)
def test_heat_index_preset_requires_hi_column(preset):
    with pytest.raises(ValueError, match="'hi'.*not found"):
        SummaryPlot(pd.DataFrame({"other": [32]})).set_regions(preset=preset)


def test_heat_index_alias_is_ambiguous():
    with pytest.raises(
        ValueError, match="Unknown preset.*heat_index_rothfusz.*heat_index_schoen"
    ):
        SummaryPlot(pd.DataFrame({"hi": [32]})).set_regions(preset="hi")


@pytest.mark.parametrize("preset", HI_PRESETS)
def test_heat_index_rounded_outputs_and_original_categories(preset):
    df = pd.DataFrame({"hi": [32.0, 32.0]})
    numeric = SummaryPlot(df).set_regions(preset=preset).plot()
    assert numeric.percentages["caution"] == 100
    original = (
        SummaryPlot(df)
        .set_categories(
            ["caution", "extreme caution"],
            labels=HI_LABELS,
            colors=["green", "yellow", "orange", "red", "black"],
        )
        .plot()
    )
    assert original.percentages.tolist() == [0, 50, 50, 0, 0]
    unrounded = (
        SummaryPlot(pd.DataFrame({"hi": [31.99, 32.01]}))
        .set_regions(preset=preset)
        .plot()
    )
    pd.testing.assert_series_equal(
        unrounded.percentages, original.percentages, check_names=False
    )
