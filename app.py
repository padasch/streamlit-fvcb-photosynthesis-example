from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None

DARK2_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
DARK_GRID = "#d3d3d3"
DARK_ZERO = "#4b5563"


def _resilience_sim_signature(
    predictor: str,
    response_var: str,
    trajectory_steps: int,
    diagnostic_window: int,
    indicator_window: int,
    enable_lag: bool,
    enable_drift: bool,
    forcing_white_noise: bool,
    resilience_2d_mode: bool,
    secondary_predictor: str,
    profile: dict,
    trajectory_settings: list[dict],
    fluctuation_scale: float,
    mean_reversion: float,
    step_factor: float,
    seed: int,
    frame_speed_ms: int,
    lag_steps_global: int,
    lag_mode: str,
    lag_reference: float,
    lag_sensitivity: float,
):
    """Build a stable cache key for resilience simulation state."""
    trajectory_signature = tuple(
        (
            float(settings["mean_value"]),
            float(settings["drift_start"]),
            float(settings["drift_end"]),
            float(settings["lag_steps"]),
            str(settings.get("lag_mode", "Fixed lag")),
            float(settings.get("lag_reference", 0.0)),
            float(settings.get("lag_sensitivity", 1.0)),
            float(settings.get("secondary_value", 0.0))
            if np.isfinite(float(settings.get("secondary_value", 0.0)))
            else -1.0e12,
        )
        for settings in trajectory_settings
    )
    profile_signature = tuple(
        (
            key,
            float(profile[key]) if isinstance(profile[key], (int, float, np.floating, np.integer)) else int(profile[key]),
        )
        for key in (
            "par",
            "ci",
            "tleaf",
            "vpd",
            "vcmax25",
            "jmax25",
            "tpu",
            "tpu_enabled",
            "rd25",
            "alpha",
            "theta",
            "temp_optimum_enabled",
            "temp_optimum_c",
            "temp_optimum_width_c",
            "vpd_half",
            "vpd_exp",
            "eavc",
            "eaj",
            "eagamma",
            "eakc",
            "eako",
            "eard",
            "gamma25",
            "kc25",
            "ko25",
            "o2",
        )
    )

    return (
        str(predictor),
        str(response_var),
        int(diagnostic_window),
        int(indicator_window),
        int(trajectory_steps),
        int(enable_lag),
        int(enable_drift),
        int(bool(forcing_white_noise)),
        int(bool(resilience_2d_mode)),
        str(secondary_predictor),
        float(fluctuation_scale),
        float(mean_reversion),
        float(step_factor),
        int(seed),
        int(frame_speed_ms),
        int(lag_steps_global),
        str(lag_mode),
        float(lag_reference),
        float(lag_sensitivity),
        trajectory_signature,
        profile_signature,
    )

st.set_page_config(
    page_title="FvCB playground",
    page_icon="🌿",
    layout="wide",
)


def _is_dark_theme():
    """Return True when Streamlit is using a dark theme."""
    theme_base = st.get_option("theme.base")
    return str(theme_base).lower() == "dark"


def ensure_session_state_initialized():
    """Ensure all app state used across both pages is initialized."""
    if not st.session_state.get("app_initialized"):
        reset_all_settings()
        st.session_state["saved_curves"] = []
        st.session_state["next_curve_id"] = 1
        st.session_state["app_initialized"] = True


def get_app_page():
    """Return the selected app page from compact top controls."""
    options = ["Photosynthesis", "Resilience"]
    if hasattr(st, "segmented_control"):
        return st.sidebar.segmented_control(
            "View",
            options=options,
            default="Photosynthesis",
            key="app_page",
            label_visibility="collapsed",
        )
    return st.sidebar.radio(
        "View",
        options=options,
        index=0,
        horizontal=True,
        key="app_page_legacy",
    )


def _build_centered_chart_container(display_width: float):
    """Return chart container tuple plus index so display is centered."""
    width = float(display_width)
    if not np.isfinite(width):
        width = 75.0
    width = float(np.clip(width, 0.0, 100.0))
    if width >= 99.999:
        return (st.container(),), 0

    width = max(width, 1.0)
    side = (100.0 - width) / 2.0
    columns = st.columns([side, width, side], gap="small")
    return columns, 1


def _trajectory_colors(count: int):
    return [DARK2_PALETTE[idx % len(DARK2_PALETTE)] for idx in range(count)]


def _blend_hex_color_with_white(color: str, white_fraction: float) -> str:
    """Blend a hex color toward white; white_fraction=0 keeps the original color."""
    color = str(color).strip()
    if not color.startswith("#") or len(color) != 7:
        return color
    white_fraction = float(np.clip(white_fraction, 0.0, 1.0))
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    red = int(round(red + (255 - red) * white_fraction))
    green = int(round(green + (255 - green) * white_fraction))
    blue = int(round(blue + (255 - blue) * white_fraction))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _trajectory_time_gradient_color(color: str, segment_index: int, max_segment_index: int) -> str:
    """Return a light-to-dark color for trajectory segment order."""
    if max_segment_index <= 0:
        return color
    time_fraction = float(segment_index) / float(max_segment_index)
    return _blend_hex_color_with_white(color, white_fraction=0.72 * (1.0 - time_fraction))


def build_environment_trajectories(
    base_value: float,
    fluctuation_scale: float,
    steps: int,
    trajectories: int,
    seed: int,
    mean_reversion: float,
    step_factor: float = 0.5,
    value_min: float = -np.inf,
    value_max: float = np.inf,
) -> np.ndarray:
    """Build mean-reverting random walk trajectories around an environmental mean."""
    rng = np.random.default_rng(seed)
    trajectories_matrix = np.empty((trajectories, steps), dtype=float)
    step_sd = max(fluctuation_scale * step_factor, 0.05)
    clamp_distance = max(4.0 * fluctuation_scale, 1.0)
    lower_bound = max(value_min, base_value - clamp_distance)
    upper_bound = min(value_max, base_value + clamp_distance)
    if upper_bound <= lower_bound:
        lower_bound = value_min
        upper_bound = value_max

    for idx in range(trajectories):
        walk = np.full(steps, base_value, dtype=float)
        walk[0] = np.clip(
            base_value + rng.normal(0.0, step_sd),
            lower_bound,
            upper_bound,
        )
        for step in range(1, steps):
            mean_term = mean_reversion * (base_value - walk[step - 1])
            walk[step] = walk[step - 1] + mean_term + rng.normal(0.0, step_sd)
            walk[step] = np.clip(walk[step], lower_bound, upper_bound)
        trajectories_matrix[idx] = walk

    return trajectories_matrix


def rolling_window_variance(values: np.ndarray, window: int) -> np.ndarray:
    """Return trailing-window sample variance for each trajectory."""
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    for trajectory_idx in range(values.shape[0]):
        series = values[trajectory_idx]
        for step in range(window - 1, values.shape[1]):
            chunk = series[step - window + 1 : step + 1]
            chunk = chunk[np.isfinite(chunk)]
            if chunk.size >= 2:
                result[trajectory_idx, step] = np.var(chunk, ddof=1)
    return result


def rolling_window_lag1_autocorrelation(values: np.ndarray, window: int) -> np.ndarray:
    """Return trailing-window lag-1 autocorrelation for each trajectory."""
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    for trajectory_idx in range(values.shape[0]):
        series = values[trajectory_idx]
        for step in range(window - 1, values.shape[1]):
            chunk = series[step - window + 1 : step + 1]
            previous = chunk[:-1]
            current = chunk[1:]
            finite = np.isfinite(previous) & np.isfinite(current)
            if finite.sum() < 3:
                continue
            previous = previous[finite]
            current = current[finite]
            if np.std(previous) == 0.0 or np.std(current) == 0.0:
                continue
            result[trajectory_idx, step] = np.corrcoef(previous, current)[0, 1]
    return result


def _kendall_tau_p(series_x: np.ndarray, series_y: np.ndarray) -> tuple[float, float]:
    """Compute Kendall tau and two-sided p-value with scipy when available, else fallback."""
    x = np.asarray(series_x, dtype=float)
    y = np.asarray(series_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        return (np.nan, np.nan)
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan)
    x = x[mask]
    y = y[mask]
    if kendalltau is not None:
        try:
            result = kendalltau(x, y)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            pass

    n = x.size
    if n < 3:
        return (np.nan, np.nan)

    s = 0.0
    for i in range(n - 1):
        delta_x = x[i + 1 :] - x[i]
        delta_y = y[i + 1 :] - y[i]
        product = delta_x * delta_y
        s += np.sum(np.sign(product))

    tau = s / (0.5 * n * (n - 1))

    _, tie_counts = np.unique(y, return_counts=True)
    tie_counts = tie_counts[tie_counts > 1]
    var_s = n * (n - 1) * (2 * n + 5)
    if tie_counts.size:
        var_s -= np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    var_s = var_s / 18.0
    if not np.isfinite(var_s) or var_s <= 0:
        return float(tau), np.nan

    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    return float(tau), p_value


def kendall_mann_trend_results(values: np.ndarray, time_axis: np.ndarray, trajectory_names: list[str] | None = None):
    """Return per-trajectory (name, tau, p-value) tuples for diagnostics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return []
    trajectory_count = values.shape[0]
    default_names = [f"Trajectory {idx + 1}" for idx in range(trajectory_count)]
    if trajectory_names is None:
        names = default_names
    else:
        names = list(trajectory_names)[:trajectory_count]
        if len(names) < trajectory_count:
            names.extend(default_names[len(names) : trajectory_count])
        names = [str(name).strip() or default_names[idx] for idx, name in enumerate(names)]

    results = []
    for idx in range(trajectory_count):
        tau, p_value = _kendall_tau_p(time_axis, values[idx])
        results.append(
            {
                "name": names[idx],
                "tau": float(tau),
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    return results


def apply_first_order_lag(
    response_values: np.ndarray,
    lag_steps: int,
    predictor_values: np.ndarray | None = None,
    lag_reference: float | None = None,
    lag_sensitivity: float = 1.0,
    use_distance_weight: bool = False,
) -> np.ndarray:
    """Apply a first-order lag to a 1D trajectory with optional distance-weighted lag.

    The distance weighting is based on how far predictor values are from a reference point:
    lag_k(t) = lag_steps * (1 + lag_sensitivity * |x_t - lag_reference| / distance_scale),
    where distance_scale is the maximum predictor distance to the reference.
    """
    values = np.asarray(response_values, dtype=float)
    lag_steps = max(int(lag_steps), 0)
    lagged = values.copy()

    # No lag should be exactly identity.
    if lag_steps == 0:
        return lagged

    alpha = 1.0 / (1.0 + lag_steps)

    use_distance_weight = (
        use_distance_weight
        and predictor_values is not None
        and lag_reference is not None
        and lag_sensitivity > 0
        and np.isfinite(lag_reference)
    )
    predictor_array = np.asarray(predictor_values, dtype=float) if use_distance_weight else None
    if use_distance_weight and predictor_array.shape != values.shape:
        use_distance_weight = False
    if use_distance_weight:
        finite_predictor = np.isfinite(predictor_array)
        if not finite_predictor.any():
            use_distance_weight = False
        else:
            predictor_min = float(np.nanmin(predictor_array))
            predictor_max = float(np.nanmax(predictor_array))
            reference = float(lag_reference)
            distance_scale = max(abs(reference - predictor_min), abs(predictor_max - reference))
            if not np.isfinite(distance_scale) or distance_scale <= 0:
                use_distance_weight = False

    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return lagged

    segments = []
    segment_start = finite_idx[0]
    previous = finite_idx[0]
    for current in finite_idx[1:]:
        if current == previous + 1:
            previous = current
            continue
        segments.append((segment_start, previous))
        segment_start = current
        previous = current
    segments.append((segment_start, previous))

    for start, end in segments:
        lagged[start] = values[start]
        for step in range(start + 1, end + 1):
            if use_distance_weight:
                distance = abs(float(predictor_array[step]) - float(lag_reference))
                dynamic_lag = lag_steps * (1.0 + lag_sensitivity * (distance / distance_scale))
                effective_alpha = 1.0 / (1.0 + dynamic_lag) if dynamic_lag >= 0 else alpha
            else:
                effective_alpha = alpha
            lagged[step] = lagged[step - 1] + effective_alpha * (values[step] - lagged[step - 1])

    return lagged


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
PREDICTOR_CONFIG = {
    "PAR": {
        "axis_label": "PAR (µmol m⁻² s⁻¹)",
        "unit": "µmol m⁻² s⁻¹",
        "min": 0.0,
        "max": 2400.0,
        "default": 1200.0,
        "step": 25.0,
        "amplitude": 200.0,
    },
    "C_i": {
        "axis_label": "C_i (ppm)",
        "unit": "ppm",
        "min": 20.0,
        "max": 2000.0,
        "default": 440.0,
        "step": 1.0,
        "amplitude": 60.0,
    },
    "T_leaf": {
        "axis_label": "Leaf temperature (°C)",
        "unit": "°C",
        "min": 5.0,
        "max": 45.0,
        "default": 25.0,
        "step": 0.1,
        "amplitude": 1.0,
    },
    "VPD": {
        "axis_label": "VPD (kPa)",
        "unit": "kPa",
        "min": 0.1,
        "max": 6.0,
        "default": 1.2,
        "step": 0.1,
        "amplitude": 0.2,
    },
}


def _response_unit_label(response_var):
    return f"{_response_axis_label(response_var)} (µmol m⁻² s⁻¹)"


def _predictor_unit_label(predictor):
    return PREDICTOR_CONFIG[_normalize_predictor(predictor)]["axis_label"]


def _default_trajectory_mean(predictor, trajectory_idx):
    config = PREDICTOR_CONFIG[_normalize_predictor(predictor)]
    offset = {
        "PAR": 200.0,
        "C_i": 100.0,
        "T_leaf": 5.0,
        "VPD": 0.6,
    }[_normalize_predictor(predictor)]
    value = config["default"] + trajectory_idx * offset
    return float(np.clip(value, config["min"], config["max"]))


def _default_trajectory_name(trajectory_idx: int) -> str:
    return {0: "Healthy", 1: "Stressed"}.get(trajectory_idx, f"Trajectory {trajectory_idx + 1}")


def _predictor_profile_key(predictor: str) -> str:
    """Map predictor display keys to profile keys used by FvCB evaluation."""
    return {
        "PAR": "par",
        "C_i": "ci",
        "T_leaf": "tleaf",
        "VPD": "vpd",
    }[_normalize_predictor(predictor)]


def _format_resilience_secondary_label(secondary_predictor: str, value: float) -> str:
    """Format a secondary-condition value for trajectory labels/notes."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return ""

    config = PREDICTOR_CONFIG.get(_normalize_predictor(secondary_predictor))
    if config is None:
        return f"{val}"
    unit = config["unit"]
    if secondary_predictor == "T_leaf":
        return f"{val:.1f} °C"
    if secondary_predictor == "C_i":
        return f"{val:.0f} {unit}"
    if secondary_predictor == "VPD":
        return f"{val:.2f} {unit}"
    return f"{val:.0f} {unit}"


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
    temp_leaf_c = np.asarray(temp_leaf_c, dtype=float)
    vpd = np.asarray(vpd, dtype=float)
    vpd = np.maximum(vpd, 0.01)
    ci, par, temp_leaf_c, vpd = np.broadcast_arrays(ci, par, temp_leaf_c, vpd)
    temp_leaf_k = temp_leaf_c + 273.15

    vcmax = arrhenius_25_to_t(params["vcmax25"], params["eavc"], temp_leaf_k)
    jmax = arrhenius_25_to_t(params["jmax25"], params["eaj"], temp_leaf_k)
    gamma = arrhenius_25_to_t(params["gamma25"], params["eagamma"], temp_leaf_k)
    kc = arrhenius_25_to_t(params["kc25"], params["eakc"], temp_leaf_k)
    ko = arrhenius_25_to_t(params["ko25"], params["eako"], temp_leaf_k)

    if params.get("temp_optimum_enabled", False):
        temp_optimum = float(params.get("temp_optimum_c", 25.0))
        temp_width = max(float(params.get("temp_optimum_width_c", 6.0)), 0.1)
        temp_response = np.exp(-((temp_leaf_c - temp_optimum) / temp_width) ** 2)
        vcmax *= temp_response
        jmax *= temp_response

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
        "temp_optimum_enabled": st.session_state["temp_optimum_enabled"],
        "temp_optimum_c": st.session_state["temp_optimum_c"],
        "temp_optimum_width_c": st.session_state["temp_optimum_width_c"],
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
    """Build long-form rows for plotting."""
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


def build_photosynthesis_plotly_figure(
    plot_data,
    x_label: str,
    y_label: str,
    show_components: bool,
    response_var: str,
    display_height: int,
    x_range=None,
    y_range=None,
):
    if go is None or plot_data.empty:
        return None

    rate_colors = {
        "Ac": DARK2_PALETTE[1],
        "Aj": DARK2_PALETTE[3],
        "A_p": DARK2_PALETTE[5],
    }

    if show_components and response_var == "A_net":
        component_mode = True
    else:
        component_mode = False
        unique_curves = list(dict.fromkeys(plot_data["curve"].tolist()))
        curve_colors = {
            curve_name: DARK2_PALETTE[idx % len(DARK2_PALETTE)]
            for idx, curve_name in enumerate(unique_curves)
        }
    unique_curves = list(dict.fromkeys(plot_data["curve"].tolist()))

    def _curve_color(row):
        if component_mode:
            rate = str(row["rate"])
            if rate.startswith("Ac"):
                return rate_colors["Ac"]
            if rate.startswith("A_p"):
                return rate_colors["A_p"]
            if rate.startswith("Aj"):
                return rate_colors["Aj"]
            return DARK2_PALETTE[0]
        return curve_colors.get(row["curve"], DARK2_PALETTE[7])

    figure = go.Figure()
    legend_registry = set()

    for _, segment in plot_data.groupby("series_id", sort=False):
        if segment.empty:
            continue
        segment = segment.sort_values("x")
        rate = str(segment["rate"].iloc[0])
        curve = str(segment["curve"].iloc[0])
        line_style = str(segment["line_style"].iloc[0])
        is_default = bool(segment["is_default"].iloc[0])
        x_values = segment["x"].to_numpy()
        y_values = segment["value"].to_numpy()

        trace_color = _curve_color(segment.iloc[0])
        dash_style = "solid" if line_style == "net" else "dash"
        line_width = (
            2.6
            if (line_style == "net" and is_default)
            else (2.2 if line_style == "net" else 2.0)
        )
        opacity = 1.0 if is_default else 0.65

        if component_mode:
            legend_key = (rate, line_style)
            trace_name = rate
        else:
            legend_key = ("curve", curve)
            trace_name = curve
        show_legend = legend_key not in legend_registry
        if show_legend:
            legend_registry.add(legend_key)

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=trace_name,
                legendgroup=trace_name,
                showlegend=show_legend,
                line=dict(color=trace_color, width=line_width, dash=dash_style),
                opacity=opacity,
                hovertemplate="",
                hoverinfo="skip",
            )
        )

    chart_labels = [c for c in unique_curves if c != "Current defaults"]
    if len(unique_curves) >= 2:
        for curve_name in chart_labels:
            curve_rows = plot_data[plot_data["curve"] == curve_name]
            finite = np.isfinite(curve_rows["value"].to_numpy())
            if not finite.any():
                continue
            curve_tail = curve_rows.loc[finite].iloc[-1]
            figure.add_annotation(
                x=float(curve_tail["x"]),
                y=float(curve_tail["value"]),
                text=curve_name,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=6,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                borderpad=2,
            )

    figure.add_hline(
        y=0.0,
        line_color=DARK_ZERO,
        line_width=1.4,
        line_dash="dot",
    )
    figure.update_layout(
        title=f"{y_label} vs {x_label}",
        height=display_height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="black"),
        hovermode=False,
        showlegend=True,
        legend=dict(
            title=dict(
                text="Curve regime" if component_mode else "Curve",
                font=dict(color="black", size=12),
            ),
            font=dict(color="black"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            x=1.02,
            xanchor="left",
            y=1.02,
        ),
        xaxis=dict(
            title=x_label,
            title_font=dict(color="black", size=14),
            tickfont=dict(color="black", size=11),
            gridcolor=DARK_GRID,
            linecolor="#111827",
            linewidth=1.2,
            showgrid=True,
            range=x_range,
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(color="black", size=14),
            tickfont=dict(color="black", size=11),
            gridcolor=DARK_GRID,
            linecolor="#111827",
            linewidth=1.2,
            showgrid=True,
            range=y_range,
        ),
    )
    return figure


def default_curve_name(n):
    return f"Curve {n}"


def reset_all_settings():
    defaults = {
        "predictor": "PAR",
        "display_width": 75,
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
        "temp_optimum_enabled": True,
        "temp_optimum_c": 25.0,
        "temp_optimum_width_c": 6.0,
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
        "resilience_lag_steps": 1,
        "resilience_lag_default_steps": 1,
        "resilience_lag_mode": "Fixed lag",
        "resilience_lag_sensitivity": 1.0,
        "resilience_2d_mode": False,
        "resilience_2d_mode_initialized": False,
        "resilience_condition_predictor": "T_leaf",
        "resilience_forcing_mode": "white",
        "resilience_fluctuation_scale_pct": 5,
        "resilience_diagnostic_window": 30,
        "resilience_indicator_window": 30,
        "resilience_trajectory_count": 2,
        "resilience_trajectory_steps": 120,
        "resilience_frame_speed_ms": 110,
        "resilience_auto_update": True,
        "resilience_enable_lag": False,
        "resilience_enable_drift": False,
        "resilience_sim_signature": None,
        "resilience_sim_cached_payload": None,
        "resilience_model_tier": "Basic",
        "resilience_trajectory_name_0": "Healthy",
        "resilience_trajectory_name_1": "Stressed",
        "resilience_trajectory_name_2": "Trajectory 3",
        "resilience_trajectory_name_3": "Trajectory 4",
        "resilience_trajectory_name_4": "Trajectory 5",
        "resilience_trajectory_name_5": "Trajectory 6",
    }
    for trajectory_idx in range(6):
        for predictor_name in PREDICTOR_OPTIONS:
            defaults[f"resilience_condition_{predictor_name}_{trajectory_idx}"] = _default_trajectory_mean(
                predictor_name, trajectory_idx
            )
        for predictor_name in PREDICTOR_OPTIONS:
            defaults[f"resilience_lag_reference_{predictor_name}"] = (
                25.0 if predictor_name == "T_leaf" else float(PREDICTOR_CONFIG[predictor_name]["default"])
            )
        defaults[f"resilience_mean_{PREDICTOR_OPTIONS[0]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[0], trajectory_idx
        )
        defaults[f"resilience_mean_{PREDICTOR_OPTIONS[1]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[1], trajectory_idx
        )
        defaults[f"resilience_mean_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[2], trajectory_idx
        )
        defaults[f"resilience_mean_{PREDICTOR_OPTIONS[3]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[3], trajectory_idx
        )
        defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[0]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[0], trajectory_idx
        )
        defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[0]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[0], trajectory_idx
        )
        defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[1]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[1], trajectory_idx
        )
        defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[1]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[1], trajectory_idx
        )
        if PREDICTOR_OPTIONS[2] == "T_leaf" and trajectory_idx == 0:
            defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = 15.0
            defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = 25.0
        elif PREDICTOR_OPTIONS[2] == "T_leaf" and trajectory_idx == 1:
            defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = 25.0
            defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = 35.0
        else:
            defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = _default_trajectory_mean(
                PREDICTOR_OPTIONS[2], trajectory_idx
            )
            defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[2]}_{trajectory_idx}"] = _default_trajectory_mean(
                PREDICTOR_OPTIONS[2], trajectory_idx
            )
        defaults[f"resilience_drift_start_{PREDICTOR_OPTIONS[3]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[3], trajectory_idx
        )
        defaults[f"resilience_drift_end_{PREDICTOR_OPTIONS[3]}_{trajectory_idx}"] = _default_trajectory_mean(
            PREDICTOR_OPTIONS[3], trajectory_idx
        )
    defaults["resilience_condition_T_leaf_0"] = 25.0
    defaults["resilience_condition_T_leaf_1"] = 35.0
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


def build_resilience_animation_figure(
    predictor_values,
    response_values,
    predictor_anoms,
    response_anoms,
    baseline_curve_x,
    baseline_curve_response,
    predictor_label,
    response_label,
    trajectory_names,
    frame_speed_ms,
    baseline_curve_names=None,
    chart_height=1120,
):
    """Return a dual-panel Plotly animation for resilience trajectories."""
    if go is None or make_subplots is None:
        return None

    trajectory_count = predictor_values.shape[0]
    steps = predictor_values.shape[1]
    trajectory_names = list(
        trajectory_names
        if trajectory_names is not None
        else [f"Trajectory {idx + 1}" for idx in range(trajectory_count)]
    )
    trajectory_names = [
        str(name).strip() if str(name).strip() else f"Trajectory {idx + 1}"
        for idx, name in enumerate(trajectory_names[:trajectory_count])
    ]
    if len(trajectory_names) < trajectory_count:
        trajectory_names.extend(
            f"Trajectory {idx + 1}" for idx in range(len(trajectory_names), trajectory_count)
        )
    time = np.arange(steps)
    template = "plotly_white"
    neutral_color = "black"
    neutral_grid = DARK_GRID
    muted_color = "#6b7280"
    colors = _trajectory_colors(trajectory_count)
    environment_anomaly = predictor_anoms[0]

    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
        row_heights=[0.56, 0.44],
        vertical_spacing=0.17,
        subplot_titles=(
            f"{response_label} vs {predictor_label}",
            "Trajectory anomalies over time",
        ),
    )

    # Top panel: response curve and trajectory positions in predictor-response space.
    baseline_response_array = np.asarray(baseline_curve_response) if baseline_curve_response is not None else np.array([])
    if (
        isinstance(baseline_curve_response, (list, tuple))
        or (isinstance(baseline_response_array, np.ndarray) and baseline_response_array.ndim == 2)
    ):
        baseline_curves = baseline_curve_response
        if isinstance(baseline_curves, np.ndarray) and baseline_curves.ndim == 1:
            baseline_curves = [baseline_curves]
        baseline_curve_names = (
            list(baseline_curve_names)
            if baseline_curve_names is not None
            else list(trajectory_names)
        )
        if len(baseline_curve_names) == 0:
            baseline_curve_names = [f"Trajectory {idx + 1}" for idx in range(trajectory_count)]
        while len(baseline_curve_names) < len(baseline_curves):
            baseline_curve_names.append(f"Trajectory {len(baseline_curve_names) + 1}")
        for idx, baseline_y in enumerate(baseline_curves):
            if baseline_y is None:
                continue
            baseline_y = np.asarray(baseline_y, dtype=float)
            if baseline_y.size == 0:
                continue
            baseline_color = colors[idx % len(colors)]
            figure.add_trace(
                go.Scatter(
                    x=baseline_curve_x,
                    y=baseline_y,
                    mode="lines",
                    name=f"{baseline_curve_names[idx]} baseline",
                    line=dict(color=baseline_color, width=2.8, dash="dash"),
                    opacity=0.85,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    elif baseline_curve_response is not None:
        figure.add_trace(
                go.Scatter(
                    x=baseline_curve_x,
                    y=baseline_curve_response,
                    mode="lines",
                    name="Baseline curve",
                    line=dict(color=muted_color, width=3, dash="dash"),
                    opacity=0.85,
                    showlegend=False,
                ),
            row=1,
            col=1,
        )

    figure.add_trace(
            go.Scatter(
                x=time,
                y=environment_anomaly,
                mode="lines",
                name=f"Δ{predictor_label} forcing",
                line=dict(color="black", width=1.6),
                showlegend=False,
            ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[environment_anomaly[0]],
            mode="markers",
            marker=dict(
                color="white",
                size=10,
                symbol="circle",
                line=dict(color="black", width=1),
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    environment_marker_index = len(figure.data) - 1

    marker_indexes = []
    for idx in range(trajectory_count):
        color = colors[idx]
        trajectory_name = trajectory_names[idx]
        max_segment_index = max(steps - 2, 0)
        for step in range(max(steps - 1, 0)):
            segment_color = _trajectory_time_gradient_color(color, step, max_segment_index)
            figure.add_trace(
                go.Scatter(
                    x=predictor_values[idx, step : step + 2],
                    y=response_values[idx, step : step + 2],
                    mode="lines",
                    name=trajectory_name,
                    line=dict(color=segment_color, width=2.2),
                    legendgroup=trajectory_name,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        marker_trace = go.Scatter(
            x=[predictor_values[idx, 0]],
            y=[response_values[idx, 0]],
            mode="markers",
            marker=dict(
                color="white",
                size=12,
                symbol="circle",
                line=dict(color=color, width=2.5),
            ),
            showlegend=False,
        )
        figure.add_trace(marker_trace, row=1, col=1)
        left_marker_index = len(figure.data) - 1

        # Response anomaly line
        figure.add_trace(
            go.Scatter(
                x=time,
                y=response_anoms[idx],
                mode="lines",
                name=f"Δ{response_label} {trajectory_name}",
                line=dict(color=color, width=2.5),
                opacity=0.95,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        anet_anom_marker = go.Scatter(
            x=[0],
            y=[response_anoms[idx, 0]],
            mode="markers",
            marker=dict(
                color="white",
                size=10,
                symbol="circle",
                line=dict(color=color, width=2.2),
            ),
            showlegend=False,
        )
        figure.add_trace(anet_anom_marker, row=2, col=1)
        response_marker_index = len(figure.data) - 1
        marker_indexes.append((left_marker_index, response_marker_index))

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color="black",
                size=9,
                symbol="square",
            ),
            name=f"Δ{predictor_label} forcing",
            showlegend=True,
            visible="legendonly",
        ),
        row=2,
        col=1,
    )
    for idx in range(trajectory_count):
        color = colors[idx]
        trajectory_name = trajectory_names[idx]
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=color,
                    size=9,
                    symbol="square",
                ),
                name=trajectory_name,
                showlegend=True,
                visible="legendonly",
            ),
            row=2,
            col=1,
        )

    # Pre-compute marker trace indexes for stable frame updates.

    # Build frame traces for the moving points only.
    frames = []
    for step in range(steps):
        frame_data = [
            go.Scatter(
                x=[step],
                y=[environment_anomaly[step]],
                mode="markers",
                marker=dict(
                    color="white",
                    size=10,
                    symbol="circle",
                    line=dict(color=neutral_color, width=1.4),
                ),
            )
        ]
        frame_trace_indices = [environment_marker_index]
        for idx in range(trajectory_count):
            left_marker_index, response_marker_index = marker_indexes[idx]
            frame_data.append(
                go.Scatter(
                    x=[predictor_values[idx, step]],
                    y=[response_values[idx, step]],
                    mode="markers",
                    marker=dict(
                        color="white",
                        size=12,
                        symbol="circle",
                        line=dict(color=colors[idx], width=2.5),
                    ),
                )
            )
            frame_trace_indices.append(left_marker_index)
            frame_data.append(
                go.Scatter(
                    x=[step],
                    y=[response_anoms[idx, step]],
                    mode="markers",
                    marker=dict(
                        color="white",
                        size=10,
                        symbol="circle",
                        line=dict(color=colors[idx], width=2.2),
                    ),
                )
            )
            frame_trace_indices.append(response_marker_index)
        frames.append(
            go.Frame(data=frame_data, name=f"step-{step}", traces=frame_trace_indices)
        )

    slider_steps = []
    for step in range(steps):
        slider_steps.append(
            {
                "args": [
                    [f"step-{step}"],
                    {
                        "frame": {"duration": frame_speed_ms, "redraw": True},
                        "mode": "immediate",
                    },
                ],
                "label": str(step + 1),
                "method": "animate",
            }
        )

    figure.frames = frames
    left_x_values = np.concatenate(
        [np.asarray(baseline_curve_x, dtype=float).ravel(), predictor_values.ravel()]
    )
    left_y_values = np.concatenate(
        [np.asarray(baseline_curve_response, dtype=float).ravel(), response_values.ravel()]
    )
    anomaly_values = np.concatenate(
        [
            np.asarray(response_anoms, dtype=float).ravel(),
            np.asarray(environment_anomaly, dtype=float).ravel(),
        ]
    )
    finite_left_x = left_x_values[np.isfinite(left_x_values)]
    finite_left_y = left_y_values[np.isfinite(left_y_values)]
    finite_anomaly_y = anomaly_values[np.isfinite(anomaly_values)]
    left_x_range = None
    left_y_range = None
    anomaly_y_range = None
    if finite_left_x.size:
        left_x_range = [float(np.min(finite_left_x)), float(np.max(finite_left_x))]
    if finite_left_y.size:
        y_min = float(np.min(finite_left_y))
        y_max = float(np.max(finite_left_y))
        y_pad = max((y_max - y_min) * 0.06, 0.5)
        left_y_range = [y_min - y_pad, y_max + y_pad]
    if finite_anomaly_y.size:
        y_min = float(np.min(finite_anomaly_y))
        y_max = float(np.max(finite_anomaly_y))
        y_pad = max((y_max - y_min) * 0.06, 0.5)
        anomaly_y_range = [y_min - y_pad, y_max + y_pad]

        figure.update_layout(
        height=chart_height,
        template=template,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="black"),
        hovermode=False,
        legend=dict(
            orientation="h",
            y=-0.24,
            x=0.0,
            xanchor="left",
            yanchor="top",
            traceorder="normal",
            itemwidth=130,
            valign="top",
            title=dict(text="", font=dict(color="black", size=12)),
            font=dict(color="black", size=12),
            itemsizing="constant",
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=20, r=30, t=220, b=170),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.28,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_speed_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "transition": {"duration": 0},
                                "frame": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.0,
                "y": 1.17,
                "xanchor": "left",
                "yanchor": "top",
                "currentvalue": {"prefix": "Step: "},
                "transition": {"duration": 0},
                "pad": {"b": 26, "t": 10},
                "steps": slider_steps,
            }
        ],
    )

    figure.update_xaxes(
        title_text=predictor_label,
        range=left_x_range,
        showgrid=True,
        gridcolor=neutral_grid,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text=response_label,
        range=left_y_range,
        showgrid=True,
        gridcolor=neutral_grid,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Time step",
        showgrid=True,
        gridcolor=neutral_grid,
        range=[0.0, float(steps - 1)] if steps > 1 else [0.0, 1.0],
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text=f"Anomalies: Δ{predictor_label}, Δ{response_label}",
        range=anomaly_y_range,
        showgrid=True,
        gridcolor=neutral_grid,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
        row=2,
        col=1,
    )
    figure.add_hline(
        y=0,
        line_color=DARK_ZERO,
        line_dash="dot",
        line_width=1.4,
        opacity=0.5,
        row=2,
        col=1,
        layer="below",
    )
    return figure


def build_resilience_setpoint_curve_figure(
    setpoint_axis,
    setpoint_curves,
    trajectory_names,
    secondary_values,
    secondary_responses,
    setpoint_label,
    response_label,
    chart_height=460,
    trajectory_colors=None,
):
    """Return a response-vs-setpoint chart for 2D resilience comparisons."""
    if go is None:
        return None

    setpoint_axis = np.asarray(setpoint_axis, dtype=float)
    if setpoint_axis.size == 0:
        return None

    setpoint_curves = np.asarray(setpoint_curves, dtype=float)
    if setpoint_curves.ndim != 2:
        setpoint_curves = np.asarray(setpoint_curves, dtype=float).reshape((1, -1))

    trajectory_count = min(len(trajectory_names), int(setpoint_curves.shape[0]))
    trajectory_names = [
        str(trajectory_names[idx]).strip() if str(trajectory_names[idx]).strip() else f"Trajectory {idx + 1}"
        for idx in range(trajectory_count)
    ]
    if trajectory_colors is None:
        trajectory_colors = _trajectory_colors(trajectory_count)
    else:
        trajectory_colors = list(trajectory_colors)
        if len(trajectory_colors) < trajectory_count:
            trajectory_colors = _trajectory_colors(trajectory_count)

    figure = go.Figure()
    for idx in range(trajectory_count):
        response_curve = np.asarray(setpoint_curves[idx], dtype=float)
        if response_curve.size == 0 or not np.isfinite(response_curve).any():
            continue

        response_curve = np.asarray(response_curve, dtype=float)
        color = trajectory_colors[idx % len(trajectory_colors)]
        secondary_value = float(secondary_values[idx]) if idx < len(secondary_values) else np.nan
        secondary_response = float(secondary_responses[idx]) if idx < len(secondary_responses) else np.nan

        figure.add_trace(
            go.Scatter(
                x=setpoint_axis,
                y=response_curve,
                mode="lines",
                name=trajectory_names[idx],
                line=dict(color=color, width=3.0, dash="dash"),
                opacity=0.88,
                showlegend=True,
            )
        )

        if np.isfinite(secondary_value) and np.isfinite(secondary_response):
            figure.add_trace(
                go.Scatter(
                    x=[secondary_value],
                    y=[secondary_response],
                    mode="markers+text",
                    text=[f"{trajectory_names[idx]}<br>{secondary_value:.1f}"],
                    textposition="top center",
                    name=f"{trajectory_names[idx]} setpoint",
                    marker=dict(color="white", size=10, line=dict(color=color, width=2)),
                    textfont=dict(color=color, size=12),
                    showlegend=False,
                )
            )

    finite_y = np.concatenate([
        curve.ravel()[np.isfinite(curve.ravel())]
        for curve in setpoint_curves[:trajectory_count]
        if np.asarray(curve).size
    ], axis=0)
    finite_y = np.asarray(finite_y, dtype=float)
    finite_x = np.asarray(setpoint_axis, dtype=float)[np.isfinite(setpoint_axis)]

    x_range = [float(np.min(finite_x)), float(np.max(finite_x))] if finite_x.size else None
    y_range = None
    if finite_y.size:
        y_pad = max(0.06 * (float(np.nanmax(finite_y)) - float(np.nanmin(finite_y))), 0.5)
        y_range = [float(np.nanmin(finite_y)) - y_pad, float(np.nanmax(finite_y)) + y_pad]

    figure.update_layout(
        height=chart_height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="black", size=12),
        title=f"Setpoint response curves ({setpoint_label})",
        title_font=dict(color="black", size=14),
        margin=dict(l=20, r=20, t=55, b=35),
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.15,
            xanchor="left",
            yanchor="top",
            font=dict(color="black", size=11),
            itemsizing="constant",
        ),
    )
    figure.update_xaxes(
        title_text=setpoint_label,
        range=x_range,
        showgrid=True,
        gridcolor=DARK_GRID,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
    )
    figure.update_yaxes(
        title_text=response_label,
        range=y_range,
        showgrid=True,
        gridcolor=DARK_GRID,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
    )
    if finite_y.size:
        figure.add_hline(
            y=0,
            line_color=DARK_ZERO,
            line_dash="dot",
            line_width=1.3,
            opacity=0.5,
        )
    return figure


def build_resilience_indicator_figure(
    time,
    values,
    title,
    y_title,
    colors,
    trajectory_names=None,
    figure_height=380,
    show_zero_line=True,
    show_mean_line=False,
    trend_stats=None,
    environment_series=None,
    environment_trend_stats=None,
    environment_name="Environmental factor",
):
    """Return a compact Plotly line chart for rolling resilience indicators."""
    figure = go.Figure()
    template = "plotly_white"
    neutral_color = "#374151"
    grid_color = DARK_GRID
    diagnostic_palette = _trajectory_colors(values.shape[0])
    default_names = [f"Trajectory {idx + 1}" for idx in range(values.shape[0])]
    if trajectory_names is None:
        trajectory_names = default_names
    else:
        trajectory_names = list(trajectory_names)[:values.shape[0]]
        if len(trajectory_names) < values.shape[0]:
            trajectory_names.extend(default_names[len(trajectory_names) : values.shape[0]])
        trajectory_names = [str(name).strip() if str(name).strip() else default_names[idx] for idx, name in enumerate(trajectory_names)]
    for idx in range(values.shape[0]):
        figure.add_trace(
            go.Scatter(
                x=time,
                y=values[idx],
                mode="lines",
                name=trajectory_names[idx],
                line=dict(color=diagnostic_palette[idx % len(diagnostic_palette)], width=3.2),
                opacity=1.0,
                showlegend=True,
            )
        )

    if show_mean_line:
        finite_counts = np.isfinite(values).sum(axis=0)
        sums = np.nansum(values, axis=0)
        mean_values = np.divide(
            sums,
            finite_counts,
            out=np.full(values.shape[1], np.nan, dtype=float),
            where=finite_counts > 0,
        )
        figure.add_trace(
            go.Scatter(
                x=time,
                y=mean_values,
                mode="lines",
                name="Mean",
                line=dict(color=neutral_color, width=4.5, dash="dash"),
                showlegend=False,
            )
        )

    if environment_series is not None:
        env = np.asarray(environment_series, dtype=float).reshape(-1)
        if env.size > len(time):
            env = env[: len(time)]
        elif env.size < len(time):
            env = np.pad(
                env,
                (0, len(time) - env.size),
                mode="constant",
                constant_values=np.nan,
            )

        finite_values = np.isfinite(values)
        if np.any(finite_values):
            value_min = float(np.nanmin(values[finite_values]))
            value_max = float(np.nanmax(values[finite_values]))
            value_mid = 0.5 * (value_min + value_max)
            value_half_range = max(1e-12, 0.5 * max(abs(value_max - value_min), 1.0))
        else:
            value_mid = 0.0
            value_half_range = 1.0

        env_mask = np.isfinite(env)
        if np.any(env_mask):
            env_min = float(np.nanmin(env[env_mask]))
            env_max = float(np.nanmax(env[env_mask]))
            if env_max > env_min:
                env_norm = (env - env_min) / (env_max - env_min)
            else:
                env_norm = np.full_like(env, 0.5)
            env_scaled = value_mid + value_half_range * (env_norm - 0.5) * 0.9
            figure.add_trace(
                go.Scatter(
                    x=time,
                    y=env_scaled,
                    mode="lines",
                    name=environment_name,
                    line=dict(color="black", width=1.4),
                    showlegend=False,
                )
            )

    if trend_stats is None:
        trend_stats = []
    else:
        trend_stats = list(trend_stats)
    if environment_trend_stats:
        trend_stats.extend(environment_trend_stats)

    if trend_stats:
        for idx, item in enumerate(trend_stats):
            name = str(item.get("name", "Trajectory"))
            tau = item.get("tau", np.nan)
            p_value = item.get("p_value", np.nan)
            if np.isfinite(tau) and np.isfinite(p_value):
                p_decimal = f"{p_value:.2f}"
                p_scientific = f"{p_value:.2e}"
                line_text = f"{name}: τ = {tau:.3f}, p = {p_decimal} ({p_scientific})"
            elif np.isfinite(tau):
                line_text = f"{name}: τ = {tau:.3f}, p = N/A"
            else:
                line_text = f"{name}: τ = N/A, p = N/A"
            is_significant = np.isfinite(tau) and np.isfinite(p_value) and p_value < 0.05
            if is_significant:
                line_text = f"<b>{line_text}</b>"
            line_color = diagnostic_palette[idx % len(diagnostic_palette)]
            if idx >= values.shape[0]:
                line_color = "black"
            figure.add_annotation(
                text=line_text,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.99 - (0.085 * idx),
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                font=dict(color=line_color, size=12),
                borderwidth=0,
                borderpad=0,
            )

    figure.update_layout(
        title=title,
        height=figure_height,
        template=template,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="black"),
        hovermode=False,
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(
            orientation="v",
            y=1.08,
            x=1.02,
            xanchor="left",
            yanchor="bottom",
            title=dict(
                text="Legend",
                font=dict(size=11, color="black"),
            ),
            font=dict(size=11, color="black"),
        ),
    )
    time_min = float(time[0]) if len(time) else 0.0
    time_max = float(time[-1]) if len(time) else 1.0
    figure.update_xaxes(
        title_text="Time step",
        showgrid=True,
        gridcolor=grid_color,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
        range=[time_min, time_max],
    )
    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=grid_color,
        title_font=dict(color="black", size=14),
        tickfont=dict(color="black", size=11),
        linecolor="#111827",
        linewidth=1.2,
    )
    if show_zero_line:
        figure.add_hline(y=0, line_color=DARK_ZERO, line_width=1.8)
    return figure


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

        By default, `V_cmax` and `J_max` also receive a simple peaked temperature response centered near 25 °C. This keeps the teaching example close to the familiar optimum-style `A_net`-temperature curve while still exposing the Arrhenius controls.

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
           - `Ac`: raw Rubisco-limited biochemical potential
           - `Ac-limited A_net`: net assimilation is controlled by `Ac`
           - `Aj`: raw electron transport-limited biochemical potential
           - `Aj-limited A_net`: net assimilation is controlled by `Aj`
           - `A_p`: raw TPU-limited biochemical potential
           - `A_p-limited A_net`: net assimilation is controlled by `A_p`
           - color mapping: all regimes use the ColorBrewer Dark2 palette
        5. Compare where the transition moves when you edit one parameter.
        """
    )


def render_resilience_page():
    """Render the resilience page with fluctuating temperature trajectories and anomalies."""
    if go is None or make_subplots is None:
        st.error("Plotly is required for the resilience animation.")
        st.info("Please add `plotly>=5.x` to your environment or requirements.")
        return

    st.title("Resilience")
    st.caption(
        "Simulate fluctuating environmental trajectories and compare their model responses "
        "against each trajectory's own mean baseline."
    )

    if st.session_state.get("resilience_fluctuation_mode") == "Percent of baseline":
        st.session_state["resilience_fluctuation_mode"] = "Percent of mean"
    if st.session_state.get("resilience_fluctuation_mode") == "Absolute °C":
        st.session_state["resilience_fluctuation_mode"] = "Absolute units"
    if st.session_state.get("resilience_fluctuation_mode") not in (
        "Absolute units",
        "Percent of mean",
        None,
    ):
        st.session_state["resilience_fluctuation_mode"] = "Absolute units"
    if st.session_state.get("resilience_predictor") not in PREDICTOR_OPTIONS:
        st.session_state["resilience_predictor"] = "T_leaf"
    if st.session_state.get("resilience_response_var") not in RESPONSE_OPTIONS:
        st.session_state["resilience_response_var"] = "A_net"
    legacy_tier = st.session_state.get("resilience_model_tier", "Basic")
    if legacy_tier not in ("Basic", "Lag response", "Lag + drift"):
        legacy_tier = "Basic"
    if "resilience_enable_lag" not in st.session_state:
        st.session_state["resilience_enable_lag"] = legacy_tier != "Basic"
    if "resilience_enable_drift" not in st.session_state:
        st.session_state["resilience_enable_drift"] = legacy_tier == "Lag + drift"
    enable_lag = bool(st.session_state.get("resilience_enable_lag", False))
    enable_drift = bool(st.session_state.get("resilience_enable_drift", False))

    resilience_2d_mode = bool(st.session_state.get("resilience_2d_mode", False))
    resilience_condition_predictor = st.session_state.get("resilience_condition_predictor", "T_leaf")
    if resilience_2d_mode and not st.session_state.get("resilience_2d_mode_initialized", False):
        st.session_state["resilience_predictor"] = "PAR"
        resilience_condition_predictor = "T_leaf"
        st.session_state["resilience_condition_predictor"] = resilience_condition_predictor
        st.session_state["resilience_condition_T_leaf_0"] = 25.0
        st.session_state["resilience_condition_T_leaf_1"] = 35.0
        st.session_state["resilience_2d_mode_initialized"] = True

    resilience_2d_mode = bool(st.session_state.get("resilience_2d_mode", False))

    trajectory_count = int(st.session_state.get("resilience_trajectory_count", 2))
    trajectory_steps = int(st.session_state.get("resilience_trajectory_steps", 120))
    predictor = _normalize_predictor(st.session_state.get("resilience_predictor", "T_leaf"))
    response_var = st.session_state.get("resilience_response_var", "A_net")
    predictor_config = PREDICTOR_CONFIG[predictor]
    default_mean_key = f"resilience_default_mean_{predictor}"
    default_mean_value = float(
        st.session_state.get(default_mean_key, predictor_config["default"])
    )
    fluctuation_scale_mode = st.session_state.get("resilience_fluctuation_mode", "Absolute units")
    fluctuation_scale_pct = int(st.session_state.get("resilience_fluctuation_scale_pct", 5))
    amplitude_key = f"resilience_fluctuation_scale_{predictor}"
    if fluctuation_scale_mode == "Absolute units":
        fluctuation_scale = float(
            st.session_state.get(amplitude_key, predictor_config["amplitude"])
        )
    else:
        fluctuation_scale = default_mean_value * (fluctuation_scale_pct / 100.0)
    mean_reversion = float(st.session_state.get("resilience_mean_reversion", 0.35))
    step_factor = float(st.session_state.get("resilience_step_factor", 0.55))
    forcing_mode = str(st.session_state.get("resilience_forcing_mode", "white"))
    if forcing_mode not in ("white", "mean_reverting"):
        forcing_mode = "white"
        st.session_state["resilience_forcing_mode"] = forcing_mode
    forcing_white_noise = forcing_mode == "white"
    seed = int(st.session_state.get("resilience_seed", 42))
    frame_speed_ms = int(st.session_state.get("resilience_frame_speed_ms", 110))
    diagnostic_window = int(st.session_state.get("resilience_diagnostic_window", 30))
    indicator_window = int(st.session_state.get("resilience_indicator_window", diagnostic_window))
    if trajectory_steps <= 0:
        trajectory_steps = 120
        st.session_state["resilience_trajectory_steps"] = trajectory_steps
    diagnostic_window = max(5, min(diagnostic_window, max(5, trajectory_steps)))
    indicator_window = max(5, min(indicator_window, max(5, trajectory_steps)))
    st.session_state["resilience_diagnostic_window"] = diagnostic_window
    st.session_state["resilience_indicator_window"] = indicator_window
    
    def _apply_resilience_preset(
        disturbance_predictor: str,
        condition_predictor: str,
        first_condition: float,
        second_condition: float,
    ) -> None:
        st.session_state["resilience_2d_mode"] = True
        st.session_state["resilience_predictor"] = disturbance_predictor
        st.session_state["resilience_condition_predictor"] = condition_predictor
        st.session_state["resilience_2d_mode_initialized"] = True
        st.session_state[f"resilience_condition_{condition_predictor}_0"] = float(first_condition)
        st.session_state[f"resilience_condition_{condition_predictor}_1"] = float(second_condition)
        if st.session_state.get("resilience_trajectory_count", 2) >= 2:
            st.session_state["resilience_trajectory_name_0"] = "Healthy"
            st.session_state["resilience_trajectory_name_1"] = "Stressed"

    def _apply_resilience_1d_preset() -> None:
        st.session_state["resilience_2d_mode"] = False
        st.session_state["resilience_2d_mode_initialized"] = False
        st.session_state["resilience_predictor"] = "T_leaf"
        st.session_state["resilience_response_var"] = "A_net"
        if st.session_state.get("resilience_trajectory_count", 2) >= 1:
            st.session_state["resilience_trajectory_name_0"] = "Healthy"
        if st.session_state.get("resilience_trajectory_count", 2) >= 2:
            st.session_state["resilience_trajectory_name_1"] = "Stressed"

    lag_steps_global = int(
        st.session_state.get(
            "resilience_lag_steps",
            st.session_state.get("resilience_lag_default_steps", 1),
        )
        )

    with st.sidebar:
        resilience_auto_update = st.session_state.get("resilience_auto_update", True)
        run_resilience_sim = False
        if not resilience_auto_update:
            st.caption("Automatic recalculation is off.")
            run_resilience_sim = st.button(
                "Run resilience simulation",
                key="resilience_run_simulation",
                use_container_width=True,
                type="primary",
                help="Apply current resilience settings and refresh charts.",
            )

        with st.expander("Model description", expanded=False):
            st.markdown(
                """
### Resilience simulation

- Two optional model effects are available:
  - **Lag**: first-order response lag is applied after model evaluation.
  - **Drift**: each trajectory follows a predictor trajectory that drifts linearly from start to final value.
  - A common, shared environmental anomaly is generated once and then added to each trajectory base path.
  - The default forcing mode is white noise (independent shocks, no memory). Optionally switch to mean-reverting forcing for persistent trajectories.
- Each trajectory is evaluated through the same FvCB evaluator used in the photosynthesis tab, with all non-target parameters held fixed.
- Baseline response is computed at each trajectory mean.
- Trajectory anomalies are shown as:
  `response_anomaly[t] = lagged_response[t] - baseline_response`

 - Optional lag (`Response lag`) is applied **after** model evaluation using shared lag settings for all trajectories.

  For lag steps `k`, the exponential smoothing is:

  `α = 1 / (1 + k)`

  `y_lag[0] = y_raw[0]`

  `y_lag[t] = y_lag[t-1] + α * (y_raw[t] - y_lag[t-1])`

  where `k = 0` is immediate response and larger values produce slower response.

Higher lag values smooth and delay response to forcing, while all other model physics and diagnostics remain unchanged.

When **Drift** is enabled, each trajectory follows:

- `predictor(t) = drift_start + (drift_end - drift_start) * (t / (T-1))`
- Shared anomalies are added afterward based on the selected forcing mode.
                """
            )
        with st.expander("Visual settings", expanded=False):
            width = st.session_state.get("display_width", 75)
            if not (0 <= width <= 100):
                st.session_state["display_width"] = 75
            display_width = st.slider(
                "Display width",
                min_value=0,
                max_value=100,
                value=st.session_state.get("display_width", 75),
                step=1,
                format="%d%%",
                key="display_width",
                help="Choose how much of the available page width the resilience figure should use.",
            )
            chart_container, chart_col = _build_centered_chart_container(display_width)
            display_height = st.slider(
                "Display height (px)",
                min_value=300,
                max_value=2400,
                value=750,
                step=50,
                key="display_height",
                help="Controls the overall height scale of the resilience plots.",
            )
            trajectory_count = st.slider(
                "Number of trajectories",
                min_value=1,
                max_value=6,
                value=trajectory_count,
                step=1,
                key="resilience_trajectory_count",
            )
            trajectory_steps = st.slider(
                "Trajectory length (steps)",
                min_value=40,
                max_value=360,
                value=trajectory_steps,
                step=5,
                key="resilience_trajectory_steps",
            )
            frame_speed_ms = st.slider(
                "Animation frame speed (ms)",
                min_value=20,
                max_value=800,
                value=frame_speed_ms,
                step=10,
                key="resilience_frame_speed_ms",
            )
            animation_height = int(min(1800, max(450, round(display_height * 1.5))))
            diagnostic_height = int(max(320, min(720, round(display_height * 0.52))))

        with st.expander("Setup - Disturbance", expanded=False):
            response_var = st.selectbox(
                "Target (y-axis)",
                options=RESPONSE_OPTIONS,
                key="resilience_response_var",
                format_func=_response_axis_label,
            )
            predictor = st.selectbox(
                "Fluctuating predictor (disturbance axis)",
                options=PREDICTOR_OPTIONS,
                key="resilience_predictor",
                format_func=_predictor_axis_label,
            )
            predictor = _normalize_predictor(predictor)
            predictor_config = PREDICTOR_CONFIG[predictor]

            default_mean_key = f"resilience_default_mean_{predictor}"
            default_mean_value = float(
                st.session_state.get(default_mean_key, predictor_config["default"])
            )
            default_mean_value = st.slider(
                f"Reference mean {_predictor_axis_label(predictor)} ({predictor_config['unit']})",
                min_value=float(predictor_config["min"]),
                max_value=float(predictor_config["max"]),
                value=default_mean_value,
                step=float(predictor_config["step"]),
                key=default_mean_key,
                help="Shared baseline value for the disturbance predictor before perturbations are added.",
            )

            st.caption(
                "This shared setpoint anchors all trajectories on the disturbance axis. "
                "Individual trajectory setpoints are configured per trajectory in the setpoint section."
            )

        with st.expander("Setup - Setpoint", expanded=False):
            resilience_2d_mode = st.toggle(
                "2D resilience",
                value=resilience_2d_mode,
                key="resilience_2d_mode",
                help=(
                    "Use one shared forcing predictor trajectory, while each trajectory has a "
                    "different fixed value for a second conditioning predictor."
                ),
            )
            if resilience_2d_mode:
                if not st.session_state.get("resilience_2d_mode_initialized", False):
                    st.session_state["resilience_predictor"] = "PAR"
                    predictor = "PAR"
                    predictor_config = PREDICTOR_CONFIG[predictor]
                    resilience_condition_predictor = "T_leaf"
                    st.session_state["resilience_condition_predictor"] = resilience_condition_predictor
                    st.session_state["resilience_condition_T_leaf_0"] = 25.0
                    st.session_state["resilience_condition_T_leaf_1"] = 35.0
                    st.session_state["resilience_2d_mode_initialized"] = True
                condition_options = [p for p in PREDICTOR_OPTIONS if p != predictor]
                if resilience_condition_predictor not in condition_options:
                    resilience_condition_predictor = condition_options[0]
                    st.session_state["resilience_condition_predictor"] = resilience_condition_predictor
                resilience_condition_predictor = st.selectbox(
                    "Conditioning predictor (static)",
                    options=condition_options,
                    key="resilience_condition_predictor",
                    format_func=_predictor_axis_label,
                )
                resilience_condition_predictor = _normalize_predictor(resilience_condition_predictor)
            else:
                st.session_state["resilience_2d_mode_initialized"] = False
                resilience_condition_predictor = st.session_state.get("resilience_condition_predictor", "T_leaf")

            resilience_auto_update = st.toggle(
                "Auto-update resilience simulation",
                value=st.session_state.get("resilience_auto_update", True),
                key="resilience_auto_update",
                help="When enabled, simulation recomputes continuously as controls change. Disable for click-to-apply behavior.",
            )
            enable_lag = st.toggle(
                "Enable lag",
                value=enable_lag,
                key="resilience_enable_lag",
                help="Adds post-processing response lag to each trajectory.",
            )
            enable_drift = st.toggle(
                "Enable drift",
                value=enable_drift,
                key="resilience_enable_drift",
                help="Adds a linear start-to-final drift to each trajectory baseline.",
            )

        with st.expander("Forcing statistics", expanded=False):
            amplitude_key = f"resilience_fluctuation_scale_{predictor}"
            fluctuation_scale_mode = st.selectbox(
                "Fluctuation scale",
                ["Absolute units", "Percent of mean"],
                index=0 if fluctuation_scale_mode == "Absolute units" else 1,
                key="resilience_fluctuation_mode",
            )
            if fluctuation_scale_mode == "Absolute units":
                fluctuation_scale = st.slider(
                    f"Default fluctuation amplitude ({predictor_config['unit']})",
                    min_value=float(predictor_config["step"]),
                    max_value=float((predictor_config["max"] - predictor_config["min"]) / 2.0),
                    value=float(st.session_state.get(amplitude_key, predictor_config["amplitude"])),
                    step=float(predictor_config["step"]),
                    key=amplitude_key,
                )
            else:
                fluctuation_scale_pct = st.slider(
                    "Fluctuation (% of mean)",
                    min_value=1,
                    max_value=30,
                    value=fluctuation_scale_pct,
                    step=1,
                    key="resilience_fluctuation_scale_pct",
                )
                fluctuation_scale = default_mean_value * (fluctuation_scale_pct / 100.0)
                st.caption(
                    f"Default effective amplitude: {fluctuation_scale:.2f} {predictor_config['unit']}"
                )
            forcing_mode = st.selectbox(
                "Environmental forcing memory",
                options=["white", "mean_reverting"],
                format_func=lambda value: "White noise (no memory)"
                if value == "white"
                else "Mean-reverting random walk",
                index=0 if forcing_mode == "white" else 1,
                key="resilience_forcing_mode",
                help="White noise produces independent forcing shocks each step; mean-reverting introduces persistence.",
            )
            forcing_white_noise = forcing_mode == "white"
            mean_reversion = st.slider(
                "Default mean reversion strength",
                min_value=0.0,
                max_value=0.95,
                value=mean_reversion,
                step=0.05,
                key="resilience_mean_reversion",
                disabled=forcing_white_noise,
            )
            step_factor = st.slider(
                "Default trajectory roughness",
                min_value=0.2,
                max_value=1.4,
                value=step_factor,
                step=0.05,
                key="resilience_step_factor",
                disabled=forcing_white_noise,
            )
            if forcing_white_noise:
                st.caption(
                    "Forcing memory is off by default: predictor anomaly has no autocorrelation by construction."
                )
            seed = st.number_input(
                "Default random seed",
                min_value=0,
                max_value=999999,
                value=seed,
                step=1,
                key="resilience_seed",
            )

        with st.expander("Resilience indicators", expanded=False):
            indicator_window = st.slider(
                "Variance/autocorrelation rolling window (steps)",
                min_value=5,
                max_value=max(5, trajectory_steps),
                value=min(indicator_window, max(5, trajectory_steps)),
                step=1,
                key="resilience_indicator_window",
                help=(
                    "Window size used for rolling variance and rolling autocorrelation calculations."
                ),
            )
            diagnostic_window = st.slider(
                "MK test rolling window (steps)",
                min_value=5,
                max_value=max(5, trajectory_steps),
                value=min(diagnostic_window, max(5, trajectory_steps)),
                step=1,
                key="resilience_diagnostic_window",
                help=(
                    "Window size used for rolling series used in the "
                    "Kendall-Mann trend diagnostics."
                ),
            )

        trajectory_settings = []
        lag_reference_key = f"resilience_lag_reference_{predictor}"
        lag_mode_key = f"resilience_lag_mode_{predictor}"
        lag_sensitivity_key = f"resilience_lag_sensitivity_{predictor}"
        lag_reference_default_base = (
            float(st.session_state.get("temp_optimum_c", 25.0))
            if predictor == "T_leaf"
            else float(predictor_config["default"])
        )
        lag_steps_global = int(
            st.session_state.get(
                "resilience_lag_steps",
                st.session_state.get("resilience_lag_default_steps", 1),
            )
        )
        lag_mode = st.session_state.get(
            lag_mode_key,
            st.session_state.get("resilience_lag_mode", "Fixed lag"),
        )
        lag_reference = float(
            st.session_state.get(
                lag_reference_key,
                st.session_state.get(
                    f"resilience_lag_reference_{predictor}_0",
                    lag_reference_default_base,
                ),
            )
        )
        lag_sensitivity = float(
            st.session_state.get(
                lag_sensitivity_key,
                st.session_state.get(
                    f"resilience_lag_sensitivity_{predictor}_0",
                    float(st.session_state.get("resilience_lag_sensitivity", 1.0)),
                ),
            )
        )

        with st.expander("Per-trajectory settings", expanded=False):
            trajectory_names = []
            trajectory_means = []
            trajectory_secondary_values = []
            if resilience_2d_mode:
                condition_config = PREDICTOR_CONFIG[_normalize_predictor(resilience_condition_predictor)]
                condition_unit = str(condition_config["unit"])
                condition_min = float(condition_config["min"])
                condition_max = float(condition_config["max"])
                condition_step = float(condition_config["step"])
            for idx in range(trajectory_count):
                if idx > 0:
                    st.divider()
                name_key = f"resilience_trajectory_name_{idx}"
                trajectory_name = st.text_input(
                    "Trajectory name",
                    value=str(st.session_state.get(name_key, _default_trajectory_name(idx))),
                    key=name_key,
                    placeholder="e.g., Healthy",
                )
                cleaned_name = trajectory_name.strip() or _default_trajectory_name(idx)
                trajectory_names.append(cleaned_name)
                st.markdown(f"**{cleaned_name} settings**")
                if enable_lag or enable_drift:
                    st.caption(
                        "Lag controls are configured in **Lag settings** and drift controls in **Drift settings**."
                    )
                if not enable_drift:
                    if resilience_2d_mode:
                        mean_value = float(default_mean_value)
                        st.caption(
                            f"Shared mean from **Predictor baseline**: "
                            f"{mean_value:.2f} {predictor_config['unit']}."
                        )
                    else:
                        mean_value = st.slider(
                            f"Mean {_predictor_axis_label(predictor)} ({predictor_config['unit']})",
                            min_value=float(predictor_config["min"]),
                            max_value=float(predictor_config["max"]),
                            value=float(st.session_state.get(
                                f"resilience_mean_{predictor}_{idx}",
                                _default_trajectory_mean(predictor, idx),
                            )),
                            step=float(predictor_config["step"]),
                            key=f"resilience_mean_{predictor}_{idx}",
                            help=(
                                "Trajectory-specific mean. The anomaly trajectory is added on top of this "
                                "value to create this trajectory’s forcing path."
                            ),
                        )
                    trajectory_means.append(mean_value)
                if resilience_2d_mode:
                    secondary_value = float(
                        st.slider(
                            f"{_predictor_axis_label(resilience_condition_predictor)} ({condition_unit})",
                            min_value=condition_min,
                            max_value=condition_max,
                            value=float(
                                st.session_state.get(
                                    f"resilience_condition_{resilience_condition_predictor}_{idx}",
                                    25.0
                                    if idx == 0 and resilience_condition_predictor == "T_leaf"
                                    else 35.0
                                    if idx == 1 and resilience_condition_predictor == "T_leaf"
                                    else float(condition_config["default"]),
                                )
                            ),
                            step=condition_step,
                            key=f"resilience_condition_{resilience_condition_predictor}_{idx}",
                        )
                    )
                    trajectory_secondary_values.append(secondary_value)
                else:
                    trajectory_secondary_values.append(np.nan)
    
        trajectory_drift_starts = []
        trajectory_drift_ends = []
        if enable_drift:
            with st.expander("Drift settings", expanded=False):
                for idx in range(trajectory_count):
                    if idx > 0:
                        st.divider()
                    trajectory_default = float(_default_trajectory_mean(predictor, idx))
                    if predictor == "T_leaf":
                        if idx == 0:
                            drift_default_start = 15.0
                            drift_default_end = 25.0
                        elif idx == 1:
                            drift_default_start = 25.0
                            drift_default_end = 35.0
                        else:
                            drift_default_start = trajectory_default
                            drift_default_end = trajectory_default
                        default_start = float(st.session_state.get(f"resilience_drift_start_{predictor}_{idx}", drift_default_start))
                        default_end = float(st.session_state.get(f"resilience_drift_end_{predictor}_{idx}", drift_default_end))
                    else:
                        default_start = float(st.session_state.get(f"resilience_drift_start_{predictor}_{idx}", trajectory_default))
                        default_end = float(st.session_state.get(f"resilience_drift_end_{predictor}_{idx}", trajectory_default))
                    drift_start_key = f"resilience_drift_start_{predictor}_{idx}"
                    drift_end_key = f"resilience_drift_end_{predictor}_{idx}"
                    drift_start = st.slider(
                        f"Start {_predictor_axis_label(predictor)} ({predictor_config['unit']}) (traj {idx + 1})",
                        min_value=float(predictor_config["min"]),
                        max_value=float(predictor_config["max"]),
                        value=default_start,
                        step=float(predictor_config["step"]),
                        key=drift_start_key,
                    )
                    drift_end = st.slider(
                        f"Final {_predictor_axis_label(predictor)} ({predictor_config['unit']}) (traj {idx + 1})",
                        min_value=float(predictor_config["min"]),
                        max_value=float(predictor_config["max"]),
                        value=default_end,
                        step=float(predictor_config["step"]),
                        key=drift_end_key,
                    )
                    trajectory_drift_starts.append(float(drift_start))
                    trajectory_drift_ends.append(float(drift_end))
        else:
            for idx in range(trajectory_count):
                trajectory_default = float(_default_trajectory_mean(predictor, idx))
                trajectory_drift_starts.append(
                    float(st.session_state.get(f"resilience_drift_start_{predictor}_{idx}", trajectory_default))
                )
                trajectory_drift_ends.append(
                    float(st.session_state.get(f"resilience_drift_end_{predictor}_{idx}", trajectory_default))
                )

        if enable_drift:
            trajectory_means = [
                0.5 * (trajectory_drift_starts[idx] + trajectory_drift_ends[idx])
                for idx in range(trajectory_count)
            ]
    
        if enable_lag:
            with st.expander("Lag settings", expanded=False):
                lag_steps_global = st.slider(
                    "Response lag (steps)",
                    min_value=0,
                    max_value=60,
                    value=lag_steps_global,
                    step=1,
                    key="resilience_lag_steps",
                    help="0 = immediate; higher values produce a slower response to forcing.",
                )
                lag_mode = st.selectbox(
                    "Lag mode",
                    options=["Fixed lag", "Reference-based lag"],
                    index=0 if lag_mode == "Fixed lag" else 1,
                    key=lag_mode_key,
                    help="Use reference-based lag if lag changes with stress relative to the chosen reference.",
                )
                if lag_mode == "Reference-based lag":
                    if predictor == "T_leaf":
                        lag_reference = float(st.session_state.get("temp_optimum_c", lag_reference_default_base))
                        st.caption(
                            f"Lag reference fixed to temperature optimum: {lag_reference:.2f} °C"
                        )
                    else:
                        lag_reference = st.slider(
                            "Lag reference",
                            min_value=float(predictor_config["min"]),
                            max_value=float(predictor_config["max"]),
                            value=lag_reference,
                            step=float(predictor_config["step"]),
                            key=lag_reference_key,
                            help="Higher lag when predictor moves away from this value.",
                        )
                    lag_sensitivity = st.slider(
                        "Distance sensitivity",
                        min_value=0.0,
                        max_value=5.0,
                        value=lag_sensitivity,
                        step=0.05,
                        key=lag_sensitivity_key,
                        help="Scales how strongly distance from the reference increases lag.",
                    )
                else:
                    if predictor == "T_leaf":
                        lag_reference = float(st.session_state.get("temp_optimum_c", lag_reference_default_base))
                    else:
                        lag_reference = float(st.session_state.get(lag_reference_key, lag_reference_default_base))
        else:
            lag_steps_global = 0
            lag_mode = "Fixed lag"
            lag_reference = float(st.session_state.get("temp_optimum_c", lag_reference_default_base))
    
        trajectory_settings = []
        for idx in range(trajectory_count):
            trajectory_settings.append(
                {
                    "name": trajectory_names[idx],
                    "mean_value": trajectory_means[idx],
                    "secondary_value": float(trajectory_secondary_values[idx]),
                    "secondary_predictor": resilience_condition_predictor,
                    "lag_steps": int(lag_steps_global),
                    "lag_mode": lag_mode,
                    "lag_reference": lag_reference,
                    "lag_sensitivity": lag_sensitivity,
                    "drift_start": trajectory_drift_starts[idx],
                    "drift_end": trajectory_drift_ends[idx],
                }
            )
    
        with st.expander("Model parameters", expanded=False):
            vcmax25 = st.slider("V_cmax,25 (µmol m⁻² s⁻¹)", 10.0, 250.0, 80.0, 1.0, key="vcmax25")
            jmax25 = st.slider("J_max,25 (µmol m⁻² s⁻¹)", 30.0, 400.0, 150.0, 1.0, key="jmax25")
            tpu_enabled = st.toggle(
                "Enable TPU limitation in model",
                value=False,
                key="tpu_enabled",
            )
            tpu = st.slider(
                "TPU capacity (µmol m⁻² s⁻¹)",
                0.0,
                120.0,
                15.0,
                1.0,
                key="tpu",
                disabled=not tpu_enabled,
            )
            rd25 = st.slider("R_d,25 (µmol m⁻² s⁻¹)", 0.0, 8.0, 1.3, 0.1, key="rd25")
            alpha = st.slider("Quantum yield α", 0.01, 0.40, 0.24, 0.01, key="alpha")
            theta = st.slider("Curvature θ", 0.20, 0.99, 0.70, 0.01, key="theta")
            eavc = st.slider("E_vc (Vcmax activation energy)", 40000, 120000, 65000, 500, key="eavc")
            eaj = st.slider("E_j (Jmax activation energy)", 20000, 120000, 50000, 500, key="eaj")
            eagamma = st.slider(
                "E_γ (Gamma* activation energy)",
                20000,
                100000,
                37830,
                500,
                key="eagamma",
            )
            eakc = st.slider("E_kc (Kc activation energy)", 50000, 120000, 79430, 500, key="eakc")
            eako = st.slider("E_ko (Ko activation energy)", 20000, 140000, 36380, 500, key="eako")
            eard = st.slider("E_Rd (Rd activation energy)", 20000, 70000, 46390, 500, key="eard")
            temp_optimum_enabled = st.toggle(
                "Temperature optimum response",
                value=True,
                key="temp_optimum_enabled",
                help="Apply a simple peaked temperature response to V_cmax and J_max.",
            )
            temp_optimum_c = st.slider(
                "Optimum temperature (°C)",
                10.0,
                40.0,
                25.0,
                0.5,
                key="temp_optimum_c",
                disabled=not temp_optimum_enabled,
            )
            temp_optimum_width_c = st.slider(
                "Temperature optimum width (°C)",
                2.0,
                20.0,
                6.0,
                0.5,
                key="temp_optimum_width_c",
                disabled=not temp_optimum_enabled,
                help="Lower values create a sharper peak around the optimum.",
            )
            vpd_half = st.slider("VPD_50 (kPa)", 0.2, 10.0, 2.0, 0.1, key="vpd_half")
            vpd_exp = st.slider("VPD sensitivity exponent", 0.3, 3.0, 1.3, 0.05, key="vpd_exp")
            gamma25 = st.slider("Γ* at 25°C (ppm)", 20.0, 100.0, 42.75, 0.25, key="gamma25")
            kc25 = st.slider("K_c at 25°C (ppm)", 100.0, 700.0, 404.9, 1.0, key="kc25")
            ko25 = st.slider(
                "K_o at 25°C (µmol mol⁻¹)",
                100000.0,
                500000.0,
                278000.0,
                100.0,
                key="ko25",
            )
            o2 = st.slider("O2 (µmol mol⁻¹)", 180000.0, 260000.0, 210000.0, 1000.0, key="o2")

        with st.expander("Environmental conditions", expanded=False):
            par = float(st.session_state.get("par", PREDICTOR_CONFIG["PAR"]["default"]))
            ci = float(st.session_state.get("ci", PREDICTOR_CONFIG["C_i"]["default"]))
            tleaf = float(st.session_state.get("tleaf", PREDICTOR_CONFIG["T_leaf"]["default"]))
            vpd = float(st.session_state.get("vpd", PREDICTOR_CONFIG["VPD"]["default"]))
            if predictor == "PAR":
                par = default_mean_value
                st.caption("PAR is controlled by the predictor trajectories.")
            else:
                par = st.slider("PAR (µmol m⁻² s⁻¹)", 0, 2400, 1200, 25, key="par")
            if predictor == "C_i":
                ci = default_mean_value
                st.caption("C_i is controlled by the predictor trajectories.")
            else:
                ci = st.slider("C_i (ppm)", 20, 2000, 440, 1, key="ci")
            if predictor == "T_leaf":
                tleaf = default_mean_value
                st.caption("T_leaf is controlled by the predictor trajectories.")
            else:
                tleaf = st.slider("Leaf temperature (°C)", 5.0, 50.0, 25.0, 0.5, key="tleaf")
            if predictor == "VPD":
                vpd = default_mean_value
                st.caption("VPD is controlled by the predictor trajectories.")
            else:
                vpd = st.slider("VPD (kPa)", 0.1, 6.0, 1.2, 0.1, key="vpd")

        st.button(
            "Reset all settings to defaults",
            on_click=reset_all_settings,
            use_container_width=True,
            type="primary",
            key="resilience_reset_all_settings",
        )

        st.markdown("### Quick presets")
        st.markdown("#### Default 1D graphs")
        st.button(
            "1D: Shared disturbance only",
            key="resilience_preset_default_1d",
            use_container_width=True,
            help="Return to 2D mode off with shared disturbance (no trajectory conditioning).",
            on_click=_apply_resilience_1d_preset,
        )
        st.markdown("#### Default 2D graphs")
        preset_col_1, preset_col_2 = st.columns(2)
        with preset_col_1:
            st.button(
                "PAR fluctuations @ 25/35 °C",
                key="resilience_preset_par_25_35",
                use_container_width=True,
                help=(
                    "Shared PAR disturbance; trajectory T_leaf setpoints at 25 °C and 35 °C."
                ),
                on_click=_apply_resilience_preset,
                args=("PAR", "T_leaf", 25.0, 35.0),
            )
        with preset_col_2:
            st.button(
                "T_leaf fluctuations @ 300/900 PAR",
                key="resilience_preset_tleaf_300_900",
                use_container_width=True,
                help=(
                    "Shared T_leaf disturbance; trajectory PAR setpoints at 300 and 900."
                ),
                on_click=_apply_resilience_preset,
                args=("T_leaf", "PAR", 300.0, 900.0),
            )

    profile = {
        "par": par,
        "ci": ci,
        "tleaf": tleaf,
        "vpd": vpd,
        "vcmax25": vcmax25,
        "jmax25": jmax25,
        "tpu_enabled": tpu_enabled,
        "tpu": tpu,
        "rd25": rd25,
        "alpha": alpha,
        "theta": theta,
        "temp_optimum_enabled": temp_optimum_enabled,
        "temp_optimum_c": temp_optimum_c,
        "temp_optimum_width_c": temp_optimum_width_c,
        "vpd_half": vpd_half,
        "vpd_exp": vpd_exp,
        "eavc": eavc,
        "eaj": eaj,
        "eagamma": eagamma,
        "eakc": eakc,
        "eako": eako,
        "eard": eard,
        "gamma25": gamma25,
        "kc25": kc25,
        "ko25": ko25,
        "o2": o2,
    }
    if not tpu_enabled:
        profile["tpu"] = 0.0

    trajectory_display_names = []
    for settings in trajectory_settings:
        trajectory_name = settings["name"]
        if resilience_2d_mode:
            sec_label = _format_resilience_secondary_label(
                settings.get("secondary_predictor", resilience_condition_predictor),
                settings.get("secondary_value", np.nan),
            )
            trajectory_name = (
                f"{trajectory_name} | {resilience_condition_predictor}={sec_label}"
                if sec_label
                else trajectory_name
            )
        trajectory_display_names.append(trajectory_name)

    requested_signature = _resilience_sim_signature(
        predictor,
        response_var,
        trajectory_steps,
        diagnostic_window,
        indicator_window,
        enable_lag,
        enable_drift,
        forcing_white_noise,
        resilience_2d_mode,
        resilience_condition_predictor,
        profile,
        trajectory_settings,
        fluctuation_scale,
        mean_reversion,
        step_factor,
        seed,
        frame_speed_ms,
        lag_steps_global,
        lag_mode,
        lag_reference,
        lag_sensitivity,
    )
    cached_signature = st.session_state.get("resilience_sim_signature")
    cached_payload = st.session_state.get("resilience_sim_cached_payload")
    signature_changed = cached_signature != requested_signature
    should_run_simulation = resilience_auto_update or run_resilience_sim or cached_payload is None
    if signature_changed:
        if resilience_auto_update:
            st.info("Calculating trajectories for the updated settings...")
        else:
            st.info(
                "Resilience settings changed but live recalculation is disabled. Click **Run resilience simulation** "
                "to refresh the curves."
            )
    if should_run_simulation:
        spinner_label = (
            "Calculating trajectories..."
            if signature_changed or cached_payload is None
            else "Running resilience simulation..."
        )
        with st.spinner(spinner_label):
            trajectory_means = np.asarray(
                [settings["mean_value"] for settings in trajectory_settings],
                dtype=float,
            )
            if enable_drift:
                time_axis = np.linspace(0.0, 1.0, trajectory_steps, dtype=float)
                trajectory_baselines = np.array(
                    [
                        settings["drift_start"]
                        + (settings["drift_end"] - settings["drift_start"]) * time_axis
                        for settings in trajectory_settings
                    ],
                    dtype=float,
                )
            else:
                trajectory_baselines = trajectory_means[:, None] + np.zeros(
                    (trajectory_count, trajectory_steps), dtype=float
                )

            lower_room = float(np.min(trajectory_baselines - predictor_config["min"]))
            upper_room = float(np.min(predictor_config["max"] - trajectory_baselines))
            shared_anomaly_limit = min(lower_room, upper_room)
            if shared_anomaly_limit <= 0:
                shared_anomaly = np.zeros(trajectory_steps, dtype=float)
            else:
                if forcing_white_noise:
                    rng = np.random.default_rng(seed)
                    shared_anomaly = rng.normal(
                        loc=0.0, scale=fluctuation_scale, size=trajectory_steps
                    )
                    shared_anomaly = np.clip(
                        shared_anomaly,
                        -shared_anomaly_limit,
                        shared_anomaly_limit,
                    )
                else:
                    shared_anomaly = build_environment_trajectories(
                        base_value=0.0,
                        fluctuation_scale=fluctuation_scale,
                        steps=trajectory_steps,
                        trajectories=1,
                        seed=seed,
                        mean_reversion=mean_reversion,
                        step_factor=step_factor,
                        value_min=-shared_anomaly_limit,
                        value_max=shared_anomaly_limit,
                    )[0]
            shared_anomaly = shared_anomaly - np.nanmean(shared_anomaly)
            shared_anomaly_mean = float(np.nanmean(shared_anomaly))
            trajectory_values = trajectory_baselines + shared_anomaly[None, :]
            curve_min = min(float(predictor_config["min"]), float(np.nanmin(trajectory_values)))
            curve_max = max(float(predictor_config["max"]), float(np.nanmax(trajectory_values)))
            curve_x = build_x_axis(predictor, curve_min, curve_max, 500)
            trajectory_response = np.empty_like(trajectory_values, dtype=float)
            setpoint_axis = None
            setpoint_curves = None
            setpoint_secondary_response = None
            if resilience_2d_mode:
                setpoint_axis = build_x_axis(
                    resilience_condition_predictor,
                    PREDICTOR_CONFIG[resilience_condition_predictor]["min"],
                    PREDICTOR_CONFIG[resilience_condition_predictor]["max"],
                    500,
                )
                setpoint_curves = np.empty((trajectory_count, setpoint_axis.size), dtype=float)
                setpoint_secondary_response = np.full((trajectory_count,), np.nan, dtype=float)
                baseline_response = np.empty((trajectory_count,), dtype=float)
                baseline_curve_response = np.empty(
                    (trajectory_count, curve_x.size),
                    dtype=float,
                )
                disturbance_profile_key = _predictor_profile_key(predictor)
                for idx in range(trajectory_count):
                    profile_i = dict(profile)
                    secondary_value = trajectory_settings[idx].get("secondary_value", np.nan)
                    secondary_predictor = trajectory_settings[idx].get(
                        "secondary_predictor",
                        resilience_condition_predictor,
                    )
                    secondary_profile_key = _predictor_profile_key(secondary_predictor)
                    profile_i[disturbance_profile_key] = float(trajectory_means[idx])
                    if np.isfinite(secondary_value):
                        profile_i[secondary_profile_key] = float(secondary_value)
                    trajectory_response[idx] = evaluate_curve_with_response(
                        predictor,
                        trajectory_values[idx],
                        profile_i,
                        response_var,
                    )
                    baseline_response[idx] = evaluate_curve_with_response(
                        predictor,
                        trajectory_means[idx],
                        profile_i,
                        response_var,
                    )
                    baseline_curve_response[idx] = evaluate_curve_with_response(
                        predictor,
                        curve_x,
                        profile_i,
                        response_var,
                    )
                    setpoint_curves[idx] = evaluate_curve_with_response(
                        resilience_condition_predictor,
                        setpoint_axis,
                        profile_i,
                        response_var,
                    )
                    if np.isfinite(secondary_value):
                        setpoint_secondary_response[idx] = float(
                            evaluate_curve_with_response(
                                resilience_condition_predictor,
                                np.array([float(secondary_value)], dtype=float),
                                profile_i,
                                response_var,
                            )[0]
                        )
            else:
                trajectory_response[:] = evaluate_curve_with_response(
                    predictor,
                    trajectory_values,
                    profile,
                    response_var,
                )
                baseline_response = evaluate_curve_with_response(
                    predictor,
                    trajectory_means,
                    profile,
                    response_var,
                )
                baseline_curve_response = evaluate_curve_with_response(
                    predictor,
                    curve_x,
                    profile,
                    response_var,
                )
            for idx in range(trajectory_count):
                lag_k = int(trajectory_settings[idx]["lag_steps"])
                if lag_k <= 0:
                    continue
                trajectory_response[idx] = apply_first_order_lag(
                    trajectory_response[idx],
                    lag_k,
                    predictor_values=(
                        trajectory_values[idx]
                        if trajectory_settings[idx].get("lag_mode", "Fixed lag") != "Fixed lag"
                        else None
                    ),
                    lag_reference=float(trajectory_settings[idx].get("lag_reference", trajectory_means[idx])),
                    lag_sensitivity=float(trajectory_settings[idx].get("lag_sensitivity", 1.0)),
                    use_distance_weight=(
                        trajectory_settings[idx].get("lag_mode", "Fixed lag") != "Fixed lag"
                    ),
                )
            predictor_anoms = trajectory_values - trajectory_means[:, None]
            response_anoms = trajectory_response - baseline_response[:, None]

            finite_trajectory = np.isfinite(trajectory_response)
            if not finite_trajectory.any():
                st.error("No finite response values were produced with these settings. Relax constraints.")
                return

            effective_diagnostic_window = max(5, min(diagnostic_window, trajectory_steps))
            effective_indicator_window = max(5, min(indicator_window, trajectory_steps))
            rolling_variance = rolling_window_variance(
                response_anoms,
                effective_indicator_window,
            )
            rolling_autocorrelation = rolling_window_lag1_autocorrelation(
                response_anoms,
                effective_indicator_window,
            )

            st.session_state["resilience_sim_signature"] = requested_signature
            st.session_state["resilience_sim_cached_payload"] = {
                "predictor": predictor,
                "response_var": response_var,
                "resilience_2d_mode": resilience_2d_mode,
                "resilience_condition_predictor": resilience_condition_predictor,
                "trajectory_settings": trajectory_settings.copy(),
                "trajectory_display_names": trajectory_display_names,
                "trajectory_count": trajectory_count,
                "trajectory_means": trajectory_means,
                "trajectory_values": trajectory_values,
                "trajectory_response": trajectory_response,
                "predictor_anoms": predictor_anoms,
                "response_anoms": response_anoms,
                "curve_x": curve_x,
                "curve_y": baseline_curve_response,
                "shared_anomaly_mean": shared_anomaly_mean,
                "baseline_response": baseline_response,
                "rolling_variance": rolling_variance,
                "rolling_autocorrelation": rolling_autocorrelation,
                "frame_speed_ms": frame_speed_ms,
                "seed": seed,
                "fluctuation_scale": fluctuation_scale,
                "trajectory_steps": trajectory_steps,
                "enable_lag": enable_lag,
                "enable_drift": enable_drift,
                "curve_min": curve_min,
                "curve_max": curve_max,
                "indicator_window": effective_indicator_window,
                "diagnostic_window": effective_diagnostic_window,
                "setpoint_axis": setpoint_axis,
                "setpoint_curves": setpoint_curves,
                "setpoint_secondary_response": setpoint_secondary_response,
                "signature": requested_signature,
            }
        sim_data = st.session_state["resilience_sim_cached_payload"]
    else:
        if not isinstance(cached_payload, dict):
            st.warning("No cached resilience result is available yet. Run the simulation once.")
            return
        sim_data = cached_payload

    trajectory_settings = sim_data["trajectory_settings"]
    trajectory_display_names = sim_data.get(
        "trajectory_display_names",
        [settings["name"] for settings in trajectory_settings],
    )
    resilience_2d_mode = bool(sim_data.get("resilience_2d_mode", False))
    resilience_condition_predictor = sim_data.get("resilience_condition_predictor", resilience_condition_predictor)
    trajectory_count = int(sim_data["trajectory_count"])
    trajectory_values = sim_data["trajectory_values"]
    trajectory_response = sim_data["trajectory_response"]
    predictor_anoms = sim_data["predictor_anoms"]
    response_anoms = sim_data["response_anoms"]
    curve_x = sim_data["curve_x"]
    curve_y = sim_data["curve_y"]
    rolling_variance = sim_data["rolling_variance"]
    rolling_autocorrelation = sim_data["rolling_autocorrelation"]
    shared_anomaly_mean = float(sim_data["shared_anomaly_mean"])
    frame_speed_ms = int(sim_data["frame_speed_ms"])
    trajectory_steps = int(sim_data["trajectory_steps"])
    fluctuation_scale = float(sim_data["fluctuation_scale"])
    seed = int(sim_data["seed"])
    enable_lag = bool(sim_data["enable_lag"])
    enable_drift = bool(sim_data["enable_drift"])
    predictor = sim_data["predictor"]
    response_var = sim_data["response_var"]
    diagnostic_window = int(sim_data.get("diagnostic_window", diagnostic_window))
    indicator_window = int(sim_data.get("indicator_window", diagnostic_window))
    setpoint_axis = sim_data.get("setpoint_axis")
    setpoint_curves = sim_data.get("setpoint_curves")
    setpoint_secondary_response = sim_data.get("setpoint_secondary_response")
    predictor_config = PREDICTOR_CONFIG[predictor]
    predictor_axis_label = _predictor_unit_label(predictor)
    response_axis_label = _response_unit_label(response_var)
    chart_container, chart_col = _build_centered_chart_container(display_width)

    with st.container():
        st.subheader("Trajectory simulation")
        if enable_drift:
            if enable_lag:
                st.caption(
                    "Mode: **Lag + drift**. Shared forcing anomalies are lagged and each trajectory drifts "
                    "linearly from its start to final predictor value."
                )
            else:
                st.caption(
                    "Mode: **Drift only**. Shared forcing anomalies are applied to trajectories whose predictors "
                    "drift linearly from start to final value."
                )
        elif enable_lag:
            st.caption(
                "Mode: **Lag only**. Shared forcing anomalies are applied with first-order lag."
            )
        else:
            st.caption("Mode: **Basic**. Trajectories are shared-anomaly mean paths without lag or drift.")
        st.caption(
            f"Shared predictor anomaly is mean-centered: average Δ{_predictor_axis_label(predictor)} "
            f"= {shared_anomaly_mean:.4f}."
        )
        figure = build_resilience_animation_figure(
            predictor_values=trajectory_values,
            response_values=trajectory_response,
            predictor_anoms=predictor_anoms,
            response_anoms=response_anoms,
            baseline_curve_x=curve_x,
            baseline_curve_response=curve_y,
            predictor_label=predictor_axis_label,
            response_label=response_axis_label,
            trajectory_names=trajectory_display_names,
            frame_speed_ms=frame_speed_ms,
            chart_height=animation_height,
            baseline_curve_names=trajectory_display_names,
        )
        if figure is None:
            st.warning("Animation figure could not be created. Verify Plotly availability.")
        else:
            with chart_container[chart_col]:
                st.plotly_chart(figure, use_container_width=True)
                st.caption(
                    f"Anomalies panel: common anomaly axis with dotted zero baselines at "
                    f"Δ{predictor_axis_label}=0 and Δ{response_axis_label}=0; "
                    f"solid shared Δ{predictor_axis_label} forcing and solid colored Δ{response_axis_label} "
                    f"trajectories with current-state markers."
                )
                if resilience_2d_mode and setpoint_axis is not None and setpoint_curves is not None:
                    secondary_values = np.asarray(
                        [settings.get("secondary_value", np.nan) for settings in trajectory_settings],
                        dtype=float,
                    )
                    secondary_responses = np.asarray(
                        setpoint_secondary_response
                        if setpoint_secondary_response is not None
                        else np.full(trajectory_count, np.nan),
                        dtype=float,
                    )
                    setpoint_figure = build_resilience_setpoint_curve_figure(
                        setpoint_axis=setpoint_axis,
                        setpoint_curves=setpoint_curves,
                        trajectory_names=trajectory_display_names,
                        secondary_values=secondary_values,
                        secondary_responses=secondary_responses,
                        setpoint_label=_predictor_unit_label(resilience_condition_predictor),
                        response_label=response_axis_label,
                        chart_height=diagnostic_height,
                        trajectory_colors=_trajectory_colors(trajectory_count),
                    )
                    if setpoint_figure is not None:
                        st.divider()
                        st.plotly_chart(setpoint_figure, use_container_width=True)
                        st.caption(
                            f"Setpoint response curves at fixed disturbance baseline for "
                            f"{_predictor_axis_label(resilience_condition_predictor)}; markers indicate each trajectory's "
                            f"configured setpoint."
                        )

    summary_rows = []
    for idx in range(trajectory_count):
        row = {
            "Trajectory": trajectory_display_names[idx],
            f"Mean {predictor_axis_label}": trajectory_settings[idx]["mean_value"],
            f"Shared amplitude ({predictor_config['unit']})": fluctuation_scale,
            "Shared seed": seed,
            f"Start {_predictor_axis_label(predictor)}": float(trajectory_settings[idx]["drift_start"]),
            f"Final {_predictor_axis_label(predictor)}": float(trajectory_settings[idx]["drift_end"]),
            f"Peak Δ{_predictor_axis_label(predictor)}": float(np.nanmax(np.abs(predictor_anoms[idx]))),
            f"Final Δ{_predictor_axis_label(predictor)}": float(predictor_anoms[idx, -1]),
            f"Min Δ{_response_axis_label(response_var)}": float(np.nanmin(response_anoms[idx])),
            f"Max Δ{_response_axis_label(response_var)}": float(np.nanmax(response_anoms[idx])),
            f"Final Δ{_response_axis_label(response_var)}": float(response_anoms[idx, -1]),
        }
        if resilience_2d_mode:
            row[f"Condition ({_predictor_axis_label(resilience_condition_predictor)})"] = float(
                trajectory_settings[idx].get("secondary_value", np.nan)
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    st.subheader("Resilience diagnostics")
    st.caption(
        f"Variance/autocorrelation rolling window: {indicator_window} steps. "
        f"MK trend diagnostics use a {diagnostic_window}-step rolling window."
    )
    colors = _trajectory_colors(trajectory_count)
    time = np.arange(trajectory_steps)
    trajectory_names = list(trajectory_display_names)
    shared_environment_series = np.asarray(trajectory_values[0], dtype=float)
    environment_trend_stats = kendall_mann_trend_results(
        np.asarray([shared_environment_series], dtype=float),
        time,
        trajectory_names=[predictor_axis_label],
    )
    variance_trend_stats = kendall_mann_trend_results(
        rolling_variance,
        time,
        trajectory_names=trajectory_names,
    )
    autocorr_trend_stats = kendall_mann_trend_results(
        rolling_autocorrelation,
        time,
        trajectory_names=trajectory_names,
    )
    if np.isfinite(rolling_variance).any():
        with chart_container[chart_col]:
            st.plotly_chart(
                build_resilience_indicator_figure(
                    time=time,
                    values=rolling_variance,
                    trajectory_names=trajectory_names,
                    title="Rolling variance",
                    y_title=f"Variance of Δ{_response_axis_label(response_var)}",
                    colors=colors,
                    figure_height=diagnostic_height,
                    show_zero_line=False,
                    show_mean_line=False,
                    trend_stats=variance_trend_stats,
                    environment_series=shared_environment_series,
                    environment_name=f"{predictor_axis_label} timeseries",
                    environment_trend_stats=environment_trend_stats,
                ),
                use_container_width=True,
            )
    else:
        st.info("Not enough finite points to calculate rolling variance.")
    if np.isfinite(rolling_autocorrelation).any():
        with chart_container[chart_col]:
            st.plotly_chart(
                build_resilience_indicator_figure(
                    time=time,
                    values=rolling_autocorrelation,
                    trajectory_names=trajectory_names,
                    title="Rolling lag-1 autocorrelation",
                    y_title=f"Autocorrelation of Δ{_response_axis_label(response_var)}",
                    colors=colors,
                    show_zero_line=False,
                    show_mean_line=False,
                    figure_height=diagnostic_height,
                    trend_stats=autocorr_trend_stats,
                    environment_series=shared_environment_series,
                    environment_name=f"{predictor_axis_label} timeseries",
                    environment_trend_stats=environment_trend_stats,
                ),
                use_container_width=True,
            )
    else:
        st.info("Not enough finite points to calculate rolling autocorrelation.")

    st.subheader("Trajectory summary")
    st.dataframe(summary.round(3), hide_index=True, use_container_width=True)


ensure_session_state_initialized()
app_page = get_app_page()

if app_page == "Resilience":
    render_resilience_page()
    st.stop()

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
        width = st.session_state.get("display_width", 75)
        if not (0 <= width <= 100):
            st.session_state["display_width"] = 75
        display_width = st.slider(
            "Display width",
            min_value=0,
            max_value=100,
            value=st.session_state.get("display_width", 75),
            step=1,
            format="%d%%",
            key="display_width",
            help="Choose how much of the available page width this chart should use.",
        )
        chart_container, chart_col = _build_centered_chart_container(display_width)
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
        temp_optimum_enabled = st.toggle(
            "Temperature optimum response",
            value=True,
            key="temp_optimum_enabled",
            help="Apply a simple peaked temperature response to V_cmax and J_max.",
        )
        temp_optimum_c = st.slider(
            "Optimum temperature (°C)",
            min_value=10.0,
            max_value=40.0,
            value=25.0,
            step=0.5,
            disabled=not temp_optimum_enabled,
            key="temp_optimum_c",
        )
        temp_optimum_width_c = st.slider(
            "Temperature optimum width (°C)",
            min_value=2.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
            disabled=not temp_optimum_enabled,
            key="temp_optimum_width_c",
            help="Lower values create a sharper peak around the optimum.",
        )

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

    with st.expander("Educational notes", expanded=False):
        render_education_page()

    st.button(
        "Reset all settings to defaults",
        on_click=reset_all_settings,
        use_container_width=True,
        type="primary",
    )

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

            chart_container, chart_col = _build_centered_chart_container(display_width)

            if go is None:
                st.warning("Plotly is unavailable in this environment. Install Plotly for chart rendering.")
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
                    x_range = [chart_x_min, chart_x_max] if (not x_dynamic) else None
                    y_range = [chart_y_min, chart_y_max] if (not y_dynamic) else None
                    figure = build_photosynthesis_plotly_figure(
                        plot_data=plot_data,
                        x_label=x_label,
                        y_label=y_label,
                        show_components=show_components,
                        response_var=response_var,
                        display_height=display_height,
                        x_range=x_range,
                        y_range=y_range,
                    )
                    if figure is None:
                        st.warning("Could not build the Plotly chart from the current data.")
                    else:
                        with chart_container[chart_col]:
                            st.plotly_chart(figure, use_container_width=True)
                            if response_var == "A_net" and show_components:
                                if show_tpu_limitation:
                                    st.caption(
                                        "Rate colors use the ColorBrewer Dark2 palette: "
                                        "Ac, Ac-limited A_net, Aj, Aj-limited A_net, A_p, A_p-limited A_net."
                                    )
                                else:
                                    st.caption(
                                        "Rate colors use the ColorBrewer Dark2 palette: "
                                        "Ac, Ac-limited A_net, Aj, Aj-limited A_net."
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
