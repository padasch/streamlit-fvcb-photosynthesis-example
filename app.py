from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None


st.set_page_config(
    page_title="FvCB playground",
    page_icon="🌿",
    layout="wide",
)


def _is_dark_theme():
    """Return True when Streamlit is using a dark theme."""
    theme_base = st.get_option("theme.base")
    return str(theme_base).lower() == "dark"


R = 8.314  # J mol^-1 K^-1
T_REF = 298.15  # 25 °C in Kelvin

RESPONSE_OPTIONS = [
    "A_net",
    "A_gross",
    "A_c",
    "A_j",
    "V_cmax",
    "J_max",
    "J",
    "R_d",
]
PREDICTOR_OPTIONS = [
    "PAR",
    "C_i",
    "T_leaf",
    "VPD",
]
PREDICTOR_LABEL = {
    "PAR": "Light (PAR)",
    "C_i": "C_i",
    "T_leaf": "T_leaf",
    "VPD": "VPD",
}
RESPONSE_LABEL = {
    "A_net": "A_net",
    "A_gross": "A_gross",
    "A_c": "A_c",
    "A_j": "A_j",
    "V_cmax": "V_cmax",
    "J_max": "J_max",
    "J": "Electron transport J",
    "R_d": "R_d",
}


def _response_axis_label(response_var):
    return RESPONSE_LABEL.get(response_var, response_var)


def _predictor_axis_label(predictor):
    return PREDICTOR_LABEL.get(predictor, predictor)


def _normalize_predictor(value):
    """Keep backwards compatibility with earlier axis-style state values."""
    if value in ("A_net vs Light (PAR)", "Light (PAR)"):
        return "PAR"
    if value in ("A_net vs C_i", "C_i"):
        return "C_i"
    if value in ("A_net vs T_leaf", "T_leaf", "Leaf temperature"):
        return "T_leaf"
    if value in ("A_net vs VPD", "VPD"):
        return "VPD"
    return value if value in PREDICTOR_OPTIONS else "PAR"


def fvcb_metrics(
    ci,
    par,
    temp_leaf_c,
    vpd,
    params,
):
    ci = np.asarray(ci, dtype=float)
    par = np.asarray(par, dtype=float)
    temp_leaf_k = np.asarray(temp_leaf_c, dtype=float) + 273.15
    vpd = np.asarray(vpd, dtype=float)
    vpd = np.maximum(vpd, 0.01)

    vcmax = arrhenius_25_to_t(params["vcmax25"], params["eavc"], temp_leaf_k)
    jmax = arrhenius_25_to_t(params["jmax25"], params["eaj"], temp_leaf_k)
    gamma = arrhenius_25_to_t(params["gamma25"], params["eagamma"], temp_leaf_k)
    kc = arrhenius_25_to_t(params["kc25"], params["eakc"], temp_leaf_k)
    ko = arrhenius_25_to_t(params["ko25"], params["eako"], temp_leaf_k)

    # Simple VPD stress proxy on photosynthetic machinery.
    vpd_stress = 1.0 / (1.0 + (vpd / params["vpd_half"]) ** params["vpd_exp"])
    vcmax *= vpd_stress
    jmax *= vpd_stress

    # temperature-sensitive respiration (approximate)
    rd = arrhenius_25_to_t(params["rd25"], params["eard"], temp_leaf_k)

    # Light-limited electron transport (non-rectangular hyperbola)
    e = params["theta"]
    rad = (params["alpha"] * par + jmax) ** 2 - 4 * e * params["alpha"] * par * jmax
    rad = np.maximum(rad, 0.0)
    j = (params["alpha"] * par + jmax - np.sqrt(rad)) / (2 * e)

    # Rubisco-limited and RuBP regeneration-limited assimilation
    o2 = params["o2"]
    w_c = vcmax * (ci - gamma) / (ci + kc * (1.0 + o2 / ko))
    w_j = j * (ci - gamma) / (4.0 * ci + 8.0 * gamma)
    w_p = np.full_like(w_c, np.nan, dtype=float)
    tpu_enabled = bool(params.get("tpu_enabled", False))

    if tpu_enabled and params["tpu"] > 0:
        w_p = np.full_like(w_c, 3.0 * params["tpu"], dtype=float)
    w = np.minimum(w_c, w_j)
    if np.isfinite(w_p).any():
        w = np.minimum(w, w_p)

    a_c = w_c - rd
    a_j = w_j - rd
    a_gross = w
    a_net = w - rd

    return {
        "A_net": a_net,
        "A_gross": a_gross,
        "A_c": a_c,
        "A_j": a_j,
        "A_p": np.where(np.isfinite(w_p), w_p - rd, np.full_like(a_c, np.nan)),
        "V_cmax": vcmax,
        "J_max": jmax,
        "J": j,
        "R_d": rd,
    }


def arrhenius_25_to_t(value_25, activation_energy, temp_leaf_k):
    """Arrhenius scaling from 25°C to leaf temperature."""
    return value_25 * np.exp(activation_energy * (temp_leaf_k - T_REF) / (T_REF * R * temp_leaf_k))


def fvcb_anet_for_ci(
    ci,
    par,
    temp_leaf_c,
    vpd,
    params,
):
    return fvcb_metrics(ci, par, temp_leaf_c, vpd, params)["A_net"]


def fvcb_anet_and_terms(
    ci,
    par,
    temp_leaf_c,
    vpd,
    params,
):
    """Return A_net plus raw A_c, A_j, and A_p components for limiting-term visualization."""
    metrics = fvcb_metrics(ci, par, temp_leaf_c, vpd, params)
    return metrics["A_net"], metrics["A_c"], metrics["A_j"], metrics["A_p"]


def build_x_axis(predictor, x_min, x_max, points):
    predictor = _normalize_predictor(predictor)
    if predictor == "PAR":
        return np.linspace(x_min, x_max, points)
    if predictor == "C_i":
        return np.linspace(max(x_min, 1.0), x_max, points)
    if predictor == "T_leaf":
        return np.linspace(x_min, x_max, points)
    return np.linspace(x_min, x_max, points)


def current_profile():
    return {
        "name": "Current defaults",
        "par": st.session_state["par"],
        "ci": st.session_state["ci"],
        "tleaf": st.session_state["tleaf"],
        "vpd": st.session_state["vpd"],
        "vcmax25": st.session_state["vcmax25"],
        "jmax25": st.session_state["jmax25"],
        "tpu_enabled": st.session_state["tpu_enabled"],
        "tpu": st.session_state["tpu"],
        "rd25": st.session_state["rd25"],
        "alpha": st.session_state["alpha"],
        "theta": st.session_state["theta"],
        "vpd_half": st.session_state["vpd_half"],
        "vpd_exp": st.session_state["vpd_exp"],
    }


def evaluate_curve(predictor, x_values, profile):
    predictor = _normalize_predictor(predictor)
    if predictor == "PAR":
        return fvcb_anet_for_ci(
            profile["ci"],
            profile["par"],
            profile["tleaf"],
            profile["vpd"],
            profile,
        )
    if predictor == "C_i":
        return fvcb_anet_for_ci(
            x_values,
            profile["par"],
            profile["tleaf"],
            profile["vpd"],
            profile,
        )
    if predictor == "T_leaf":
        return fvcb_anet_for_ci(
            profile["ci"],
            profile["par"],
            x_values,
            profile["vpd"],
            profile,
        )
    return fvcb_anet_for_ci(
        profile["ci"],
        profile["par"],
        profile["tleaf"],
        x_values,
        profile,
    )


def evaluate_curve_with_response(predictor, x_values, profile, response_var):
    predictor = _normalize_predictor(predictor)
    if predictor == "PAR":
        return fvcb_metrics(profile["ci"], x_values, profile["tleaf"], profile["vpd"], profile)[
            response_var
        ]
    if predictor == "C_i":
        return fvcb_metrics(x_values, profile["par"], profile["tleaf"], profile["vpd"], profile)[
            response_var
        ]
    if predictor == "T_leaf":
        return fvcb_metrics(profile["ci"], profile["par"], x_values, profile["vpd"], profile)[
            response_var
        ]

    # VPD predictor
    return fvcb_metrics(profile["ci"], profile["par"], profile["tleaf"], x_values, profile)[
        response_var
    ]


def evaluate_curve_payload(predictor, x_values, profile, response_var):
    if response_var == "A_net":
        anet, ac, aj, ap = evaluate_curve_components(predictor, x_values, profile)
        return anet, ac, aj, ap
    return evaluate_curve_with_response(predictor, x_values, profile, response_var), None, None, None


def evaluate_curve_components(predictor, x_values, profile):
    predictor = _normalize_predictor(predictor)
    if predictor == "PAR":
        return fvcb_anet_and_terms(
            profile["ci"],
            x_values,
            profile["tleaf"],
            profile["vpd"],
            profile,
        )
    if predictor == "C_i":
        return fvcb_anet_and_terms(
            x_values,
            profile["par"],
            profile["tleaf"],
            profile["vpd"],
            profile,
        )
    if predictor == "T_leaf":
        return fvcb_anet_and_terms(
            profile["ci"],
            profile["par"],
            x_values,
            profile["vpd"],
            profile,
        )

    # VPD predictor
    return fvcb_anet_and_terms(
        profile["ci"],
        profile["par"],
        profile["tleaf"],
        x_values,
        profile,
    )


def build_plot_frame(chart_plot, response_var, show_components, show_tpu_limitation):
    """Build long-form rows for Altair plotting with explicit legend-friendly fields."""
    x = chart_plot["x"].to_numpy()
    rows = []

    def _finite_runs(mask):
        padded = np.concatenate(([False], mask, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        return zip(changes[0::2], changes[1::2])

    def _append(name, values, rate, style, is_default):
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if not mask.any():
            return
        for segment_id, (start, end) in enumerate(_finite_runs(mask)):
            rows.append(
                pd.DataFrame(
                    {
                        "x": x[start:end],
                        "value": values[start:end],
                        "curve": name,
                        "rate": rate,
                        "line_style": style,
                        "series_id": f"{name}|{rate}|{style}|{segment_id}",
                        "is_default": 1 if is_default else 0,
                    }
                )
            )

    if not show_components or response_var != "A_net":
        for name in [c for c in chart_plot.columns if c != "x"]:
            if (
                name.endswith(" (A_c)")
                or name.endswith(" (A_j)")
                or name.endswith(" (A_p)")
            ):
                continue
            values = chart_plot[name]
            _append(
                name=name,
                values=values,
                rate=response_var,
                style="net",
                is_default=(name == "Current defaults"),
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    base_name = "Current defaults"
    curve_names = [base_name] + [curve["name"] for curve in st.session_state.saved_curves]
    for name in curve_names:
        is_default = name == base_name
        anet_col = name
        ac_col = f"{name} (A_c)"
        aj_col = f"{name} (A_j)"
        ap_col = f"{name} (A_p)"
        has_tpu_component = ap_col in chart_plot.columns

        if anet_col not in chart_plot.columns or ac_col not in chart_plot.columns or aj_col not in chart_plot.columns:
            continue

        anet = chart_plot[anet_col].to_numpy(dtype=float)
        ac = chart_plot[ac_col].to_numpy(dtype=float)
        aj = chart_plot[aj_col].to_numpy(dtype=float)
        if has_tpu_component:
            ap = chart_plot[ap_col].to_numpy(dtype=float)
            has_tpu_component = np.isfinite(ap).any()
        else:
            ap = np.full_like(anet, np.nan, dtype=float)
        show_tpu_component = show_tpu_limitation and has_tpu_component
        finite = np.isfinite(anet) & np.isfinite(ac) & np.isfinite(aj)
        if show_tpu_component:
            finite &= np.isfinite(ap)
        if not finite.any():
            continue

        _append(name, ac, "Ac", "background", is_default)
        _append(name, aj, "Aj", "background", is_default)
        if show_tpu_component:
            _append(name, ap, "A_p", "background", is_default)

        comp = np.stack([ac[finite], aj[finite]], axis=0)
        if show_tpu_component:
            comp = np.vstack([comp, ap[finite]])
        argmin = np.argmin(comp, axis=0)

        ac_limited = np.zeros_like(anet, dtype=bool)
        aj_limited = np.zeros_like(anet, dtype=bool)
        ac_limited[finite] = argmin == 0
        aj_limited[finite] = argmin == 1
        _append(
            name=name,
            values=np.where(ac_limited, anet, np.nan),
            rate="Ac-limited A_net",
            style="net",
            is_default=is_default,
        )
        _append(
            name=name,
            values=np.where(aj_limited, anet, np.nan),
            rate="Aj-limited A_net",
            style="net",
            is_default=is_default,
        )
        if show_tpu_component:
            ap_limited = np.zeros_like(anet, dtype=bool)
            ap_limited[finite] = argmin == 2
            _append(
                name=name,
                values=np.where(ap_limited, anet, np.nan),
                rate="A_p-limited A_net",
                style="net",
                is_default=is_default,
            )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def default_curve_name(n):
    return f"Curve {n}"


def reset_all_settings():
    defaults = {
        "predictor": "PAR",
        "display_width": 100,
        "display_height": 750,
        "par_x_min": 0,
        "par_x_max": 2200,
        "ci_x_min": 20,
        "ci_x_max": 1400,
        "tleaf_x_min": 5,
        "tleaf_x_max": 45,
        "vpd_x_min": 0.1,
        "vpd_x_max": 6.0,
        "x_dynamic": True,
        "y_dynamic": True,
        "chart_x_min_fixed": 0.0,
        "chart_x_max_fixed": 2200.0,
        "chart_y_min_fixed": -20.0,
        "chart_y_max_fixed": 60.0,
        "response_var": "A_net",
        "show_components": True,
        "par": 1200,
        "ci": 440,
        "tleaf": 25.0,
        "vpd": 1.2,
        "vcmax25": 80.0,
        "jmax25": 150.0,
        "tpu": 15.0,
        "rd25": 1.3,
        "alpha": 0.24,
        "theta": 0.7,
        "eavc": 65000,
        "eaj": 50000,
        "eagamma": 37830,
        "eakc": 79430,
        "eako": 36380,
        "eard": 46390,
        "vpd_half": 2.0,
        "vpd_exp": 1.3,
        "tpu_enabled": False,
        "show_tpu_limitation": False,
        "gamma25": 42.75,
        "kc25": 404.9,
        "ko25": 278000.0,
        "o2": 210000.0,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def remove_curves(curve_ids):
    if not curve_ids:
        return
    remaining = []
    for curve in st.session_state.saved_curves:
        if curve["id"] not in curve_ids:
            remaining.append(curve)
    st.session_state.saved_curves = remaining


def render_education_page():
    st.title("Model education")
    st.markdown(
        """
        This app is a **teaching-oriented, one-leaf FvCB playground**.  
        It evaluates one environmental sweep at a time and shows how net carbon gain (`A_net`) responds when one factor changes and the others are held fixed.

        In the current model, net photosynthesis is the minimum of two biochemical capacities (and TPU cap when enabled), then corrected for dark respiration:

        - **Rubisco-limited rate:**  $A_c = W_c - R_d$
        - **RuBP-regeneration rate:** $A_j = W_j - R_d$
        - **Net assimilation:** $A_{net} = \min(W_c, W_j, W_p) - R_d$

        where $W_p=3\cdot TPU$ when TPU is enabled (otherwise no $W_p$ cap), and the same respiration term is subtracted from each displayed raw pathway.
        """
    )

    st.markdown("### Core equations")
    st.latex(r"W_c = \frac{V_{cmax}\,(C_i-\Gamma^*)}{C_i + K_c\,(1+O/K_o)}")
    st.latex(
        r"W_j = \frac{\alpha \, PAR + J_{max}-\sqrt{(\alpha\,PAR+J_{max})^2-4\theta\alpha PAR J_{max}}}{2\theta}"
    )
    st.latex(
        r"R_d = R_{d,25}\cdot\exp\left(\frac{E_{Rd}(T_{leaf}-298.15)}{298.15RT_{leaf}}\right)"
    )

    st.markdown(
        """
        **What you should watch for while using this model**

        - the **shape change** of a curve (curvature, asymptote, slope, and whether there is a saturating plateau),
        - the **x-location of the regime switch** between `Ac` and `Aj` control,
        - and how quickly the model moves into low or negative `A_net` at stressful settings.
        """
    )

    st.markdown("### Environmental modification examples")
    st.markdown(
        """
        These examples assume your current defaults unless stated otherwise. A quick way to explore is to keep one slider fixed (the one selected as x-axis), then move one other environmental slider and re-check the curve.
        """
    )

    st.markdown("#### 1) Increase light (`PAR`) first, keep `C_i`, `T_leaf`, and `VPD` fixed")
    st.markdown(
        """
        With low irradiance, electron transport is usually limiting (`Aj` is lower than `Ac`), so the curve climbs steeply from low light and is often blue (`Aj` raw / `Aj-limited A_net`).

        As PAR rises, `J` saturates through the non-rectangular hyperbola. At some light level, `Ac` can become the tighter constraint and the displayed line will often turn red-limited for the upper light range.
        \n
        Practical read: this is the classic photosynthesis light response. If your default settings were set with moderate CO₂ and low VPD, the regime switch is usually visible as a distinct "kink" in the full curve.
        """
    )

    st.markdown("#### 2) Increase intercellular CO₂ (`C_i`) first, keep `PAR`, `T_leaf`, and `VPD` fixed")
    st.markdown(
        """
        Raising `C_i` tends to raise both `Ac` and `Aj`, but `Ac` is often more sensitive at low to intermediate CO₂ because its denominator includes `K_c(1+O/K_o)`.

        At low `C_i` the curve can be sharply limited by Rubisco demand and therefore look red early in the x-axis. At higher `C_i`, `Ac` can rise above `Aj`, making the regime more transport-limited over a larger part of the curve.

        Practical read: sweeping `C_i` is a good way to mimic CO₂ enrichment / stomatal openness shifts and observe a **moving Rubisco bottleneck**.
        """
    )

    st.markdown("#### 3) Raise leaf temperature (`T_leaf`) first, keep `PAR`, `C_i`, and `VPD` fixed")
    st.markdown(
        """
        Temperature adjusts multiple parameters through Arrhenius scaling in this app. That means both biochemical rates and respiration are temperature-sensitive.

        In this simplified implementation there is no explicit high-temperature deactivation term, so you may see roughly monotonic shifts in capacities but potentially smaller or even lower `A_net` if respiration rises faster than carbon gain in the same setting.

        Practical read: compare a lower and higher `T_leaf` comparison curve to see whether the limiting regime changes (often little for very short x-ranges, more obvious when baseline `C_i` is tight).
        """
    )

    st.markdown("#### 4) Increase VPD first, keep `PAR`, `C_i`, and `T_leaf` fixed")
    st.markdown(
        """
        VPD enters as a multiplicative stress term applied to both `Vcmax` and `Jmax`. As VPD increases, both `Ac` and `Aj` are pushed down together.

        Because both terms are scaled by the same stress proxy in this implementation, regime transitions may stay at similar x positions; the main visible effect is usually a global down-shift in curve height (and more negative values at the high end if respiration dominates).
\n
        Practical read: this mimics water-stress dampening of enzymatic capacity; use it to stress-test how robust your chosen `C_i` and light combinations are.
        """
    )

    st.markdown("### How to read limit information")
    st.markdown(
        """
        1. Enable **Show A_c and A_j curves** in Visual settings.
        2. Enable **Show TPU limitation in chart** in Model specifics.
        3. Enable curve adding for side-by-side scenarios (default settings vs saved comparison curves).
        4. Use this legend logic while reading the chart:
           - `Ac` (red): raw Rubisco-limited biochemical potential
           - `Ac-limited A_net` (light red): net assimilation is controlled by `Ac`
           - `Aj` (blue): raw electron transport-limited biochemical potential
           - `Aj-limited A_net` (light blue): net assimilation is controlled by `Aj`
           - `A_p` (dark green): raw TPU-limited biochemical potential
           - `A_p-limited A_net` (light green): net assimilation is controlled by `A_p`
        5. Compare where the transition moves when you edit one parameter.
        """
    )


st.title("FvCB playground")
st.caption(
    "Interactive FvCB-style model explorer for response curves versus Light, C_i, T_leaf, or VPD. "
    "Use this for quick scenario comparison and sensitivity checks."
)
if "saved_curves" not in st.session_state:
    st.session_state.saved_curves = []
if "next_curve_id" not in st.session_state:
    st.session_state.next_curve_id = 1

with st.sidebar:
    with st.expander("Visual settings", expanded=False):
        _curve_columns = st.columns(2)
        left_col = _curve_columns[0]
        right_col = _curve_columns[1]
        with left_col:
            response_var = st.selectbox(
                "Target (y-axis)",
                options=RESPONSE_OPTIONS,
                key="response_var",
                help="Choose the target variable shown on the y-axis.",
                format_func=_response_axis_label,
            )
        with right_col:
            if "axis" in st.session_state and "predictor" not in st.session_state:
                st.session_state["predictor"] = _normalize_predictor(st.session_state["axis"])
            if "predictor" not in st.session_state:
                st.session_state["predictor"] = "PAR"
            predictor = st.selectbox(
                "Predictor (x-axis)",
                options=PREDICTOR_OPTIONS,
                key="predictor",
                help="Choose which environmental variable is on the x-axis.",
                format_func=_predictor_axis_label,
            )
        if st.session_state.get("display_width", 100) not in (50, 75, 100):
            st.session_state["display_width"] = 100
        display_width = st.select_slider(
            "Display width",
            options=[50, 75, 100],
            value=100,
            format_func=lambda value: f"{value}%",
            key="display_width",
            help="Choose how much of the available page width this chart should use.",
        )
        display_height = st.slider(
            "Display height (px)",
            min_value=300,
            max_value=2400,
            value=750,
            step=50,
            key="display_height",
        )
        if predictor == "PAR":
            x_min = st.slider(
                "PAR min",
                min_value=0,
                max_value=1200,
                value=0,
                step=25,
                key="par_x_min",
            )
            x_max = st.slider(
                "PAR max",
                min_value=200,
                max_value=2400,
                value=2200,
                step=25,
                key="par_x_max",
            )
        elif predictor == "C_i":
            x_min = st.slider(
                "C_i min (ppm)",
                min_value=20,
                max_value=200,
                value=20,
                step=1,
                key="ci_x_min",
            )
            x_max = st.slider(
                "C_i max (ppm)",
                min_value=500,
                max_value=2000,
                value=1400,
                step=1,
                key="ci_x_max",
            )
        elif predictor == "T_leaf":
            x_min = st.slider(
                "T_leaf min (°C)",
                min_value=0,
                max_value=40,
                value=5,
                step=1,
                key="tleaf_x_min",
            )
            x_max = st.slider(
                "T_leaf max (°C)",
                min_value=10,
                max_value=50,
                value=45,
                step=1,
                key="tleaf_x_max",
            )
        else:
            x_min = st.slider(
                "VPD min (kPa)",
                min_value=0.1,
                max_value=5.0,
                value=0.1,
                step=0.1,
                key="vpd_x_min",
            )
            x_max = st.slider(
                "VPD max (kPa)",
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=0.1,
                key="vpd_x_max",
            )

        x_dynamic = st.toggle(
            "Dynamic x-axis",
            value=True,
            help="Disable for fixed x limits.",
            key="x_dynamic",
        )
        y_dynamic = st.toggle(
            "Dynamic y-axis",
            value=True,
            help="Disable for fixed y limits.",
            key="y_dynamic",
        )
        if x_dynamic:
            chart_x_min = x_min
            chart_x_max = x_max
        else:
            chart_x_min = st.number_input(
                "Fixed x-axis min",
                value=float(x_min),
                step=0.1,
                key="chart_x_min_fixed",
            )
            chart_x_max = st.number_input(
                "Fixed x-axis max",
                value=float(x_max),
                step=0.1,
                key="chart_x_max_fixed",
            )

        if y_dynamic:
            chart_y_min = None
            chart_y_max = None
        else:
            response_unit_label = _response_axis_label(response_var)
            chart_y_min = st.number_input(
                f"Fixed y-axis min ({response_unit_label})",
                value=-20.0,
                step=0.5,
                key="chart_y_min_fixed",
            )
            chart_y_max = st.number_input(
                f"Fixed y-axis max ({response_unit_label})",
                value=60.0,
                step=0.5,
                key="chart_y_max_fixed",
            )
        if response_var == "A_net":
            show_components = st.toggle(
                "Show A_c and A_j curves",
                value=True,
                help="Plot A_c and A_j background lines to show which process is limiting A_net.",
                key="show_components",
            )
        else:
            st.toggle(
                "Show A_c and A_j curves",
                value=False,
                disabled=True,
                help="Rate-limiting overlays are only available for A_net.",
                key="show_components",
            )
            show_components = False

    with st.expander("Environmental defaults", expanded=False):
        par = st.slider("PAR (µmol m⁻² s⁻¹)", min_value=0, max_value=2400, value=1200, step=25, key="par")
        ci = st.slider("C_i (ppm)", min_value=20, max_value=2000, value=440, step=1, key="ci")
        tleaf = st.slider("Leaf temperature (°C)", min_value=5.0, max_value=50.0, value=25.0, step=0.5, key="tleaf")
        vpd = st.slider("VPD (kPa)", min_value=0.1, max_value=6.0, value=1.2, step=0.1, key="vpd")

    with st.expander("Model specifics", expanded=False):
        st.subheader("Biochemistry")
        tpu_enabled = st.toggle(
            "Enable TPU limitation in model",
            value=False,
            key="tpu_enabled",
            help="Apply the 3×TPU cap inside A_net only when enabled.",
        )
        vcmax25 = st.slider(
            "V_cmax,25 (µmol m⁻² s⁻¹)", min_value=10.0, max_value=250.0, value=80.0, step=1.0, key="vcmax25"
        )
        jmax25 = st.slider(
            "J_max,25 (µmol m⁻² s⁻¹)", min_value=30.0, max_value=400.0, value=150.0, step=1.0, key="jmax25"
        )
        tpu = st.slider(
            "TPU capacity (µmol m⁻² s⁻¹)",
            min_value=0.0,
            max_value=120.0,
            value=15.0,
            step=1.0,
            disabled=not tpu_enabled,
            key="tpu",
        )
        show_tpu_limitation = st.toggle(
            "Show TPU limitation in chart",
            value=False,
            key="show_tpu_limitation",
            help="Plot raw TPU limitation curve and TPU-limited A_net segments in green",
        )
        if not tpu_enabled:
            show_tpu_limitation = False
        rd25 = st.slider("R_d,25 (µmol m⁻² s⁻¹)", min_value=0.0, max_value=8.0, value=1.3, step=0.1, key="rd25")
        alpha = st.slider("Quantum yield α", min_value=0.01, max_value=0.40, value=0.24, step=0.01, key="alpha")
        theta = st.slider("Curvature θ", min_value=0.20, max_value=0.99, value=0.70, step=0.01, key="theta")

        st.subheader("Temperature response (J/mol)")
        eavc = st.slider("E_vc (Vcmax activation energy)", 40000, 120000, 65000, 1000, key="eavc")
        eaj = st.slider("E_j (Jmax activation energy)", 20000, 120000, 50000, 1000, key="eaj")
        eagamma = st.slider("E_γ (Gamma* activation energy)", 20000, 100000, 37830, 1000, key="eagamma")
        eakc = st.slider("E_kc (Kc activation energy)", 50000, 120000, 79430, 1000, key="eakc")
        eako = st.slider("E_ko (Ko activation energy)", 20000, 140000, 36380, 1000, key="eako")
        eard = st.slider("E_Rd (Rd activation energy)", 20000, 70000, 46390, 1000, key="eard")

        st.subheader("VPD response")
        vpd_half = st.slider("VPD_50 (kPa)", min_value=0.2, max_value=10.0, value=2.0, step=0.1, key="vpd_half")
        vpd_exp = st.slider("VPD sensitivity exponent", min_value=0.3, max_value=3.0, value=1.3, step=0.05, key="vpd_exp")

        st.subheader("Constants")
        with st.expander("Advanced constants (optional)"):
            gamma25 = st.slider(
                "Γ* at 25°C (ppm)",
                min_value=20.0,
                max_value=100.0,
                value=42.75,
                step=0.25,
                key="gamma25",
            )
            kc25 = st.slider(
                "K_c at 25°C (ppm)",
                min_value=100.0,
                max_value=700.0,
                value=404.9,
                step=1.0,
                key="kc25",
            )
            ko25 = st.slider(
                "K_o at 25°C (µmol mol⁻¹)",
                min_value=100000.0,
                max_value=500000.0,
                value=278000.0,
                step=100.0,
                key="ko25",
            )
            o2 = st.slider(
                "O2 (µmol mol⁻¹)",
                min_value=180000.0,
                max_value=260000.0,
                value=210000.0,
                step=1000.0,
                key="o2",
            )

    with st.expander("Curve adding", expanded=False):
        default_new_name = default_curve_name(st.session_state.next_curve_id)
        new_name = st.text_input("Curve name", value=default_new_name)
        if st.button("Add current settings as comparison curve"):
            payload = current_profile()
            payload["id"] = st.session_state.next_curve_id
            payload["name"] = new_name if new_name.strip() else default_new_name
            payload["gamma25"] = gamma25
            payload["kc25"] = kc25
            payload["ko25"] = ko25
            payload["o2"] = o2 / 1000.0
            payload["eavc"] = eavc
            payload["eaj"] = eaj
            payload["eagamma"] = eagamma
            payload["eakc"] = eakc
            payload["eako"] = eako
            payload["eard"] = eard
            st.session_state.saved_curves.append(payload)
            st.session_state.next_curve_id += 1
            st.rerun()

        if st.session_state.saved_curves:
            remove_ids = st.multiselect(
                "Remove curves",
                options=[f"{c['name']} (id {c['id']})" for c in st.session_state.saved_curves],
            )
            if st.button("Delete selected", key="delete_curves"):
                ids_to_remove = {
                    int(label.split("id ")[1][:-1]) for label in remove_ids if "id " in label
                }
                remove_curves(ids_to_remove)
                st.rerun()
        else:
            st.caption("No saved comparison curves yet.")

    st.button("Reset all settings to defaults", on_click=reset_all_settings, use_container_width=True)
    with st.expander("Educational notes", expanded=False):
        render_education_page()

# Add advanced constants to current profile
base_profile = current_profile()
base_profile["gamma25"] = gamma25
base_profile["kc25"] = kc25
base_profile["ko25"] = ko25
base_profile["o2"] = o2
base_profile["eavc"] = eavc
base_profile["eaj"] = eaj
base_profile["eagamma"] = eagamma
base_profile["eakc"] = eakc
base_profile["eako"] = eako
base_profile["eard"] = eard

if x_max <= x_min:
    st.error("x max must be greater than x min.")
elif (not x_dynamic and chart_x_max <= chart_x_min) or (not y_dynamic and chart_y_max <= chart_y_min):
    st.error("Fixed axis bounds require max > min.")
else:
    points = 200
    x = build_x_axis(predictor, x_min, x_max, points)
    x_label = {
        "PAR": "PAR (µmol m⁻² s⁻¹)",
        "C_i": "C_i (ppm)",
        "T_leaf": "Leaf temperature (°C)",
        "VPD": "VPD (kPa)",
    }[predictor]
    y_label = f"{_response_axis_label(response_var)} (µmol m⁻² s⁻¹)"

    chart_df = pd.DataFrame({"x": x})

    # Current slider settings baseline curve
    if response_var == "A_net" and show_components:
        anet_base, ac_base, aj_base, ap_base = evaluate_curve_payload(
            predictor,
            x,
            base_profile,
            response_var,
        )
        chart_df["Current defaults (A_c)"] = ac_base
        chart_df["Current defaults (A_j)"] = aj_base
        if np.isfinite(ap_base).any():
            chart_df["Current defaults (A_p)"] = ap_base
        chart_df["Current defaults"] = anet_base
    else:
        response_base, _, _, _ = evaluate_curve_payload(predictor, x, base_profile, response_var)
        chart_df["Current defaults"] = response_base

    # Extra comparison curves (snapshots of previous settings)
    for curve in st.session_state.saved_curves:
        response_curve, ac_curve, aj_curve, ap_curve = evaluate_curve_payload(
            predictor,
            x,
            curve,
            response_var,
        )
        if response_var == "A_net" and show_components:
            chart_df[f"{curve['name']} (A_c)"] = ac_curve
            chart_df[f"{curve['name']} (A_j)"] = aj_curve
            if np.isfinite(ap_curve).any():
                chart_df[f"{curve['name']} (A_p)"] = ap_curve
        chart_df[curve["name"]] = response_curve

    chart_df = chart_df.replace([np.inf, -np.inf], np.nan)
    curve_columns = [c for c in chart_df.columns if c != "x"]
    if curve_columns:
        finite_rows = np.isfinite(chart_df[curve_columns].to_numpy()).any(axis=1)
        chart_df = chart_df.loc[finite_rows].copy()
    else:
        chart_df["x"] = []

    if chart_df.empty:
        st.error("No finite points to plot with current settings. Try relaxing parameter bounds.")
    else:
        chart_plot = chart_df.copy()
        if not x_dynamic:
            chart_plot = chart_plot[(chart_plot["x"] >= chart_x_min) & (chart_plot["x"] <= chart_x_max)]
        if chart_plot.empty:
            st.error("No points remain after applying fixed x-axis bounds.")
        else:
            if not y_dynamic:
                chart_plot[curve_columns] = chart_plot[curve_columns].clip(chart_y_min, chart_y_max)

            if display_width == 100:
                chart_container = (st.container(),)
            else:
                chart_container = st.columns([display_width, 100 - display_width], gap="small")

            if alt is None:
                st.warning(
                    "Altair is not available in this environment. "
                    "Showing a fallback chart without explicit A_c/A_j color layers."
                )
                net_cols = [
                    c
                    for c in chart_plot.columns
                    if c != "x" and " (A_c)" not in c and " (A_j)" not in c and " (A_p)" not in c
                ]
                net_chart_df = chart_plot.set_index("x")[net_cols].copy()
                net_chart_df["A = 0"] = 0.0
                with chart_container[0]:
                    st.line_chart(net_chart_df, height=display_height, use_container_width=True)
            else:
                plot_data = build_plot_frame(
                    chart_plot,
                    response_var,
                    show_components,
                    show_tpu_limitation if response_var == "A_net" else False,
                )
                if plot_data.empty:
                    st.warning("No drawable points under current limits.")
                else:
                    if show_components:
                        color_field = "rate"
                        color_legend_title = "Curve regime"
                        color_rates = [
                            "Ac",
                            "Ac-limited A_net",
                            "Aj",
                            "Aj-limited A_net",
                        ]
                        color_range = [
                            "#d32f2f",
                            "#ef9a9a",
                            "#1976d2",
                            "#90caf9",
                        ]
                        if show_tpu_limitation and response_var == "A_net":
                            color_rates.extend(["A_p", "A_p-limited A_net"])
                            color_range.extend(["#2e7d32", "#81c784"])
                        color_scale = alt.Scale(domain=color_rates, range=color_range)
                    else:
                        color_field = "curve"
                        color_legend_title = "Curve"
                        color_scale = alt.Scale(scheme="category10")

                    chart = (
                        alt.Chart(plot_data)
                        .mark_line()
                        .encode(
                            x=alt.X("x:Q", title=x_label),
                            y=alt.Y("value:Q", title=y_label),
                            color=alt.Color(
                                f"{color_field}:N",
                                title=color_legend_title,
                                scale=color_scale,
                            ),
                            strokeDash="line_style:N",
                            detail="series_id:N",
                            opacity=alt.condition(
                                alt.datum.is_default,
                                alt.value(1.0),
                                alt.value(0.6),
                            ),
                            size=alt.condition(
                                alt.datum.is_default,
                                alt.value(2.5),
                                alt.value(1.2),
                            ),
                            tooltip=["curve", "rate", "x", "value"],
                        )
                        .properties(height=display_height)
                        .interactive()
                    )
                    zero_line = alt.Chart(pd.DataFrame({"value": [0.0]})).mark_rule(
                        color="#ffffff" if _is_dark_theme() else "#000000",
                        strokeWidth=1.5,
                    ).encode(y="value:Q")
                    chart = chart + zero_line
                    with chart_container[0]:
                        st.altair_chart(chart, use_container_width=True)
                    if response_var == "A_net" and show_components:
                        if show_tpu_limitation:
                            st.caption(
                                "Colors: Ac (red), Ac-limited A_net (light red), Aj (blue), Aj-limited A_net (light blue), A_p (dark green), A_p-limited A_net (light green)."
                            )
                        else:
                            st.caption(
                                "Colors: Ac (red), Ac-limited A_net (light red), Aj (blue), Aj-limited A_net (light blue)."
                            )
                    else:
                        st.caption("Each curve color identifies the profile compared.")

    net_curve_columns = [
        c
        for c in chart_df.columns
        if c != "x" and not c.endswith(" (A_c)") and not c.endswith(" (A_j)")
    ]
    summary = (
        chart_df[net_curve_columns]
        .describe()
        .T[["min", "max"]]
        .rename(
            columns={
                "min": f"min {_response_axis_label(response_var)}",
                "max": f"max {_response_axis_label(response_var)}",
            }
        )
    )
    st.subheader(f"Curve summary (min / max {_response_axis_label(response_var)})")
    st.dataframe(summary.round(3))

    st.subheader("Saved curve default snapshots")
    if st.session_state.saved_curves:
        snapshot_rows = []
        for curve in st.session_state.saved_curves:
            snapshot_rows.append(
                {
                    "Curve": curve["name"],
                    "PAR": curve["par"],
                    "C_i": curve["ci"],
                    "T_leaf": curve["tleaf"],
                    "VPD": curve["vpd"],
                    "V_cmax,25": curve["vcmax25"],
                    "J_max,25": curve["jmax25"],
                    "TPU": curve["tpu"],
                    "R_d,25": curve["rd25"],
                }
            )
        st.dataframe(pd.DataFrame(snapshot_rows).set_index("Curve"))
    if not st.session_state.saved_curves:
        st.info("Add at least one comparison curve to display and compare additional default settings.")
