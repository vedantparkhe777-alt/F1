"""
F1 Telemetry & Strategy Studio
================================
A single-file Streamlit application for interactive Formula 1 race data
analysis powered by FastF1, Pandas, NumPy, and Matplotlib.

Prerequisites:
    pip install -r requirements.txt

    FastF1 ≥ 3.4 requires a free Formula 1 account for live-timing data.
    Set the following environment variables BEFORE launching the app:

        set F1_USERNAME=your@email.com
        set F1_PASSWORD=your_password
        streamlit run app.py

    Alternatively, create a `.env` file in this directory (see README).
"""

import datetime
import os
import warnings

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

# ── Load credentials from .streamlit/secrets.toml ─────────────────────
try:
    os.environ.setdefault("F1_USERNAME", st.secrets["F1_USERNAME"])
    os.environ.setdefault("F1_PASSWORD", st.secrets["F1_PASSWORD"])
except Exception:
    pass  # Will be caught later in sidebar_controls

# ──────────────────────────────────────────────────────────────────────
# 0.  Global Setup
# ──────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="F1 Telemetry & Strategy Studio",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enable FastF1 disk cache on first import
_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_cache")
os.makedirs(_cache_dir, exist_ok=True)
try:
    fastf1.Cache.configure(cache_dir=_cache_dir)
except Exception:
    # Fallback for older FastF1 versions
    fastf1.Cache.enable_cache(_cache_dir)

# Configure Matplotlib with the official F1 dark theme
fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

# ──────────────────────────────────────────────────────────────────────
# 1.  Dynamic Theme / CSS
# ──────────────────────────────────────────────────────────────────────
SEASON_THEMES = {
    2018: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2019: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2020: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2021: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2022: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2023: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2024: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2025: {"accent": "#FF1801", "name": "Classic F1 Red"},
    2026: {"accent": "#00F0FF", "name": "Cyberpunk Cyan"},
}

DEFAULT_THEME = {"accent": "#FF1801", "name": "Classic F1 Red"}


def inject_css(accent_color: str) -> None:
    """Inject dark glassmorphic CSS themed to the selected season."""
    st.markdown(
        f"""
        <style>
        /* ── Root variables ── */
        :root {{
            --accent: {accent_color};
            --glass-bg: rgba(17, 17, 27, 0.72);
            --glass-border: rgba(255, 255, 255, 0.08);
        }}

        /* ── Global background ── */
        .stApp {{
            background: linear-gradient(145deg, #0a0a14 0%, #0d1117 50%, #0a0a14 100%);
        }}

        /* ── Sidebar glass effect ── */
        section[data-testid="stSidebar"] {{
            background: var(--glass-bg) !important;
            border-right: 1px solid var(--glass-border);
            backdrop-filter: blur(18px);
        }}

        /* ── Cards / containers ── */
        div[data-testid="stExpander"],
        div[data-testid="stMetric"],
        .stTabs [data-baseweb="tab-panel"] {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            backdrop-filter: blur(14px);
            padding: 0.6rem;
        }}

        /* ── Tab strip accent ── */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: var(--accent) !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--accent) !important;
            font-weight: 700;
        }}

        /* ── Header glow ── */
        h1, h2, h3 {{
            text-shadow: 0 0 12px {accent_color}44;
        }}
        h1 {{
            color: var(--accent) !important;
        }}

        /* ── Selectbox / inputs ── */
        div[data-baseweb="select"] > div {{
            background-color: rgba(20,20,35,0.85) !important;
            border-color: var(--glass-border) !important;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-thumb {{
            background: var(--accent);
            border-radius: 3px;
        }}

        /* ── Metric value color ── */
        div[data-testid="stMetric"] label {{
            color: #aaa !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: var(--accent) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# 2.  Cached Data Loaders
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching calendar …", ttl=3600)
def load_schedule(year: int) -> pd.DataFrame:
    """Return the full event schedule for a given year."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule


@st.cache_resource(show_spinner="Loading session data – this may take a minute …")
def load_session(year: int, event_name: str, session_type: str):
    """Load and return a full FastF1 Session object (cached as a resource).

    We use cache_resource because the Session object is not serialisable
    but IS safely shareable across reruns.
    """
    session = fastf1.get_session(year, event_name, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    return session


# ──────────────────────────────────────────────────────────────────────
# 3.  Colour Helpers
# ──────────────────────────────────────────────────────────────────────
COMPOUND_COLORS_FALLBACK = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0072DB",
    "UNKNOWN": "#888888",
    "TEST_UNKNOWN": "#888888",
}


def compound_color(compound: str, session=None) -> str:
    """Return the display colour for a tyre compound string."""
    try:
        return fastf1.plotting.get_compound_color(compound, session=session)
    except Exception:
        return COMPOUND_COLORS_FALLBACK.get(str(compound).upper(), "#888888")


def driver_team_color(session, driver: str) -> str:
    """Best-effort team colour for a driver from the session."""
    try:
        laps = session.laps
        row = laps[laps["Driver"] == driver].iloc[0]
        return fastf1.plotting.get_team_color(row["Team"], session=session)
    except Exception:
        return "#ffffff"


# ──────────────────────────────────────────────────────────────────────
# 4.  Visualisation Builders
# ──────────────────────────────────────────────────────────────────────

def _dark_axes(fig, axes):
    """Apply dark background to a figure and its axes."""
    fig.patch.set_facecolor("#0a0a14")
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#0a0a14")
        ax.tick_params(colors="#ccc")
        for spine in ax.spines.values():
            spine.set_color("#333")


def build_telemetry_comparison(session, driver1, driver2, accent):
    """Speed / Throttle / Brake traces vs distance for two drivers."""
    laps = session.laps
    lap1 = laps.pick_drivers(driver1).pick_fastest()
    lap2 = laps.pick_drivers(driver2).pick_fastest()

    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()

    color1 = driver_team_color(session, driver1)
    color2 = driver_team_color(session, driver2)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    _dark_axes(fig, axes)

    # Speed
    axes[0].plot(tel1["Distance"], tel1["Speed"], color=color1, label=driver1, lw=1.2)
    axes[0].plot(tel2["Distance"], tel2["Speed"], color=color2, label=driver2, lw=1.2)
    axes[0].set_ylabel("Speed (km/h)", color="#ccc")
    axes[0].legend(loc="upper right", fontsize=9, facecolor="#111", edgecolor="#444", labelcolor="white")

    # Throttle
    axes[1].plot(tel1["Distance"], tel1["Throttle"], color=color1, lw=1.2)
    axes[1].plot(tel2["Distance"], tel2["Throttle"], color=color2, lw=1.2)
    axes[1].set_ylabel("Throttle %", color="#ccc")
    axes[1].set_ylim(-5, 105)

    # Brake
    axes[2].plot(tel1["Distance"], tel1["Brake"].astype(int), color=color1, lw=1.2)
    axes[2].plot(tel2["Distance"], tel2["Brake"].astype(int), color=color2, lw=1.2)
    axes[2].set_ylabel("Brake (on/off)", color="#ccc")
    axes[2].set_xlabel("Distance (m)", color="#ccc")
    axes[2].set_ylim(-0.1, 1.3)

    fig.suptitle(
        f"Telemetry Comparison  ·  {driver1} vs {driver2}",
        color=accent, fontsize=15, fontweight="bold", y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_track_speed_heatmap(session, driver, accent):
    """Circuit outline coloured by speed (plasma) using LineCollection."""
    laps = session.laps
    lap = laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_telemetry()

    x = tel["X"].values.astype(float)
    y = tel["Y"].values.astype(float)
    speed = tel["Speed"].values.astype(float)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    _dark_axes(fig, ax)

    norm = plt.Normalize(speed.min(), speed.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidths=4)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Speed (km/h)", color="#ccc", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ccc")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Track Speed Map  ·  {driver}  ·  Fastest Lap",
        color=accent, fontsize=14, fontweight="bold",
    )
    ax.autoscale_view()
    plt.tight_layout()
    return fig


def build_tire_strategy(session, accent):
    """Horizontal Gantt chart showing each driver's tyre stints."""
    laps = session.laps
    results = session.results

    laps_clean = laps[["Driver", "Stint", "Compound", "LapNumber"]].copy()
    laps_clean.dropna(subset=["Compound"], inplace=True)

    stints = (
        laps_clean.groupby(["Driver", "Stint", "Compound"])
        .count()
        .reset_index()
        .rename(columns={"LapNumber": "StintLength"})
    )

    # Order drivers by finishing position
    try:
        driver_order = list(results["Abbreviation"])
    except Exception:
        driver_order = sorted(laps["Driver"].unique())

    driver_order = [d for d in driver_order if d in stints["Driver"].values]

    fig, ax = plt.subplots(figsize=(14, max(6, len(driver_order) * 0.55)))
    _dark_axes(fig, ax)

    for driver in driver_order:
        drv_stints = stints[stints["Driver"] == driver].sort_values("Stint")
        prev = 0
        for _, row in drv_stints.iterrows():
            cc = compound_color(row["Compound"], session=session)
            ax.barh(
                y=driver,
                width=row["StintLength"],
                left=prev,
                color=cc,
                edgecolor="#222",
                linewidth=0.6,
                height=0.6,
            )
            ax.text(
                prev + row["StintLength"] / 2,
                driver,
                row["Compound"][0],
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="#111",
            )
            prev += row["StintLength"]

    ax.invert_yaxis()
    ax.set_xlabel("Lap Number", color="#ccc", fontsize=11)
    ax.set_title("Tire Strategy", color=accent, fontsize=14, fontweight="bold")
    ax.tick_params(colors="#ccc")

    # Compound legend
    compound_legend = []
    for cpd, clr in COMPOUND_COLORS_FALLBACK.items():
        if cpd in ("UNKNOWN", "TEST_UNKNOWN"):
            continue
        compound_legend.append(Patch(facecolor=clr, edgecolor="#555", label=cpd.title()))
    ax.legend(
        handles=compound_legend, loc="lower right",
        fontsize=8, facecolor="#111", edgecolor="#444", labelcolor="white", ncol=5,
    )

    plt.tight_layout()
    return fig


def build_minisector_dominance(session, driver1, driver2, accent, n_sectors=25):
    """Track map coloured by which driver is faster per mini-sector."""
    laps = session.laps
    lap1 = laps.pick_drivers(driver1).pick_fastest()
    lap2 = laps.pick_drivers(driver2).pick_fastest()

    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()

    color1 = driver_team_color(session, driver1)
    color2 = driver_team_color(session, driver2)

    total_dist = max(tel1["Distance"].max(), tel2["Distance"].max())
    sector_len = total_dist / n_sectors

    tel1 = tel1.copy()
    tel2 = tel2.copy()
    tel1["MiniSector"] = (tel1["Distance"] // sector_len).clip(upper=n_sectors - 1).astype(int)
    tel2["MiniSector"] = (tel2["Distance"] // sector_len).clip(upper=n_sectors - 1).astype(int)

    # Determine the faster driver per mini-sector (higher avg speed = faster)
    sector_winner = {}
    for i in range(n_sectors):
        avg1 = tel1.loc[tel1["MiniSector"] == i, "Speed"].mean()
        avg2 = tel2.loc[tel2["MiniSector"] == i, "Speed"].mean()
        avg1 = 0 if pd.isna(avg1) else avg1
        avg2 = 0 if pd.isna(avg2) else avg2
        sector_winner[i] = driver1 if avg1 >= avg2 else driver2

    # Build coloured track using tel1 X/Y as reference geometry
    x = tel1["X"].values.astype(float)
    y = tel1["Y"].values.astype(float)
    ms = tel1["MiniSector"].values

    fig, ax = plt.subplots(figsize=(10, 10))
    _dark_axes(fig, ax)

    for i in range(n_sectors):
        mask = ms == i
        if mask.sum() < 2:
            continue
        c = color1 if sector_winner[i] == driver1 else color2
        ax.plot(x[mask], y[mask], color=c, linewidth=5, solid_capstyle="round")

    legend_elements = [
        Patch(facecolor=color1, edgecolor="white", label=driver1),
        Patch(facecolor=color2, edgecolor="white", label=driver2),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=11, facecolor="#111", edgecolor="#444", labelcolor="white",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Mini-Sector Dominance ({n_sectors} sectors)  ·  {driver1} vs {driver2}",
        color=accent, fontsize=14, fontweight="bold",
    )
    ax.autoscale_view()
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# 5.  Sidebar & Calendar Logic
# ──────────────────────────────────────────────────────────────────────
def sidebar_controls():
    """Render all sidebar inputs and return selections."""
    st.sidebar.markdown("## 🏎️ F1 Studio Controls")

    # ── F1 Account Credentials ────────────────────────────────────
    has_creds = bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))
    if has_creds:
        st.sidebar.success("✅ F1 account connected", icon="🔑")
    else:
        with st.sidebar.expander("🔑 F1 Account (required for telemetry)", expanded=True):
            st.markdown(
                "FastF1 ≥ 3.4 requires a **free** [Formula 1 account]"
                "(https://account.formula1.com/) to fetch live-timing data.\n\n"
                "Set these env vars **before** launching Streamlit:\n"
                "```\n"
                "set F1_USERNAME=you@email.com\n"
                "set F1_PASSWORD=your_password\n"
                "streamlit run app.py\n"
                "```"
            )
            # Allow entering credentials in the sidebar for convenience
            f1_user = st.text_input("Email", key="f1_user")
            f1_pass = st.text_input("Password", type="password", key="f1_pass")
            if f1_user and f1_pass:
                os.environ["F1_USERNAME"] = f1_user
                os.environ["F1_PASSWORD"] = f1_pass
                st.success("Credentials set for this session!")
                has_creds = True

    available_years = list(range(2018, 2027))
    year = st.sidebar.selectbox("🗓️ Season", available_years, index=available_years.index(2025))

    # Apply season theme
    theme = SEASON_THEMES.get(year, DEFAULT_THEME)
    accent = theme["accent"]
    inject_css(accent)

    st.sidebar.caption(f"Theme: **{theme['name']}**")

    try:
        schedule = load_schedule(year)
    except Exception as e:
        st.sidebar.error(f"Could not load {year} schedule: {e}")
        st.stop()

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Parse session dates robustly — prefer the Race session date
    date_col = None
    for candidate in ("Session5Date", "EventDate"):
        if candidate in schedule.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = schedule.columns[-1]

    schedule_copy = schedule.copy()
    schedule_copy["_date"] = pd.to_datetime(schedule_copy[date_col], utc=True, errors="coerce")

    completed = schedule_copy[schedule_copy["_date"] < now].copy()
    upcoming = schedule_copy[schedule_copy["_date"] >= now].copy()

    # ── Upcoming races with countdown ──────────────────────────────
    if not upcoming.empty:
        with st.sidebar.expander(f"📅 Upcoming Races ({len(upcoming)})", expanded=False):
            for _, row in upcoming.iterrows():
                delta = row["_date"] - now
                days = delta.days
                name = row.get("EventName", row.get("OfficialEventName", "TBD"))
                st.markdown(f"• **{name}** — *{days}d remaining*")

    # ── Completed races dropdown ──────────────────────────────────
    if completed.empty:
        st.sidebar.warning("No completed races for this season yet.")
        st.stop()

    with st.sidebar.expander(f"✅ Completed Races ({len(completed)})", expanded=False):
        for _, row in completed.iterrows():
            name = row.get("EventName", row.get("OfficialEventName", "TBD"))
            st.markdown(f"• {name}")

    race_names = completed["EventName"].tolist()
    selected_race = st.sidebar.selectbox("🏁 Select Race", race_names, index=len(race_names) - 1)

    session_map = {
        "Race": "R",
        "Qualifying": "Q",
        "Sprint": "S",
        "Sprint Qualifying": "SQ",
        "FP1": "FP1",
        "FP2": "FP2",
        "FP3": "FP3",
    }
    session_label = st.sidebar.selectbox("📊 Session", list(session_map.keys()))
    session_code = session_map[session_label]

    return year, selected_race, session_code, session_label, accent


# ──────────────────────────────────────────────────────────────────────
# 6.  Main Application
# ──────────────────────────────────────────────────────────────────────
def main():
    year, race, session_code, session_label, accent = sidebar_controls()

    st.markdown("# 🏎️ F1 Telemetry & Strategy Studio")
    st.markdown(f"**{year}  ·  {race}  ·  {session_label}**")

    # ── Load session ──────────────────────────────────────────────
    try:
        session = load_session(year, race, session_code)
    except Exception as e:
        st.error(
            f"⚠️ Could not load telemetry for **{race} ({session_label})**.  \n"
            f"The session data may be unavailable or corrupted on the FIA servers.  \n"
            f"```{e}```"
        )
        st.stop()

    laps_df = session.laps
    drivers = sorted(laps_df["Driver"].unique())

    if len(drivers) < 2:
        st.warning("Not enough driver data available for comparison.")
        st.stop()

    # ── Driver pickers (below title, above tabs) ──────────────────
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        driver1 = st.selectbox("Driver A", drivers, index=0)
    with col_d2:
        default_idx = min(1, len(drivers) - 1)
        driver2 = st.selectbox("Driver B", drivers, index=default_idx)

    # ── Tabbed analyses ───────────────────────────────────────────
    tab_telem, tab_heatmap, tab_strat, tab_mini = st.tabs([
        "📈 Speed / Throttle / Brake",
        "🗺️ Track Speed Heatmap",
        "🔧 Tire Strategy",
        "🔬 Mini-Sector Dominance",
    ])

    # ─── Tab 1: Telemetry Comparison ──────────────────────────────
    with tab_telem:
        try:
            fig = build_telemetry_comparison(session, driver1, driver2, accent)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Telemetry comparison unavailable: {e}")

    # ─── Tab 2: Track Speed Heatmap ──────────────────────────────
    with tab_heatmap:
        heatmap_driver = st.selectbox("Select driver for heatmap", drivers, key="hm_drv")
        try:
            fig = build_track_speed_heatmap(session, heatmap_driver, accent)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Track speed heatmap unavailable: {e}")

    # ─── Tab 3: Tire Strategy ────────────────────────────────────
    with tab_strat:
        try:
            fig = build_tire_strategy(session, accent)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Tire strategy chart unavailable: {e}")

    # ─── Tab 4: Mini-Sector Dominance ────────────────────────────
    with tab_mini:
        n_sectors = st.slider("Number of mini-sectors", 10, 50, 25, step=5, key="ms_n")
        try:
            fig = build_minisector_dominance(session, driver1, driver2, accent, n_sectors)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Mini-sector analysis unavailable: {e}")


if __name__ == "__main__":
    main()
