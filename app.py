# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import fastf1
import fastf1.plotting
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --------------------------------------------------------------
# 0️⃣ Global configuration
# --------------------------------------------------------------
# Enable FastF1 cache once on startup
fastf1.Cache.enable_cache('f1_cache')
# Use FastF1's official dark theme for all Matplotlib figures
fastf1.plotting.setup_mpl()

# --------------------------------------------------------------
# 1️⃣ Helper utilities
# --------------------------------------------------------------
def get_season_accent(year: int) -> str:
    """Return a CSS hex colour for the given season."""
    # 2026 – Cyberpunk Cyan, 2025 – Classic F1 Red, others – neutral gray
    mapping = {
        2026: "#00FFFF",  # cyan
        2025: "#FF0000",  # red
    }
    return mapping.get(year, "#666666")  # fallback dark gray


def inject_css(accent: str):
    """Inject dark‑theme, glass‑morphic CSS with a dynamic accent."""
    css = f"""
    <style>
    /* Streamlit background */
    .stApp {{ background: #0e0e0e; color: #e0e0e0; }}

    /* Glass‑morphic containers */
    .glass {{
        background: rgba(20,20,20,0.4);
        border-radius: 12px;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }}

    /* Accent colour for headers / highlights */
    .accent {{ color: {accent} !important; }}
    .stButton>button {{ background: {accent}; border: none; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_event_schedule() -> pd.DataFrame:
    """Fetch the full FIA event calendar (all years) once."""
    return fastf1.get_event_schedule()


def split_schedule(df: pd.DataFrame):
    """Separate upcoming vs completed races based on UTC now."""
    now = datetime.now(timezone.utc)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    upcoming = df[df["Date"] > now].copy()
    completed = df[df["Date"] <= now].copy()
    # Add a human‑readable countdown for upcoming races
    upcoming["Countdown"] = upcoming["Date"].apply(
        lambda d: str(d - now).split(".")[0]
    )
    return upcoming, completed


@st.cache_data(show_spinner=False)
def load_session(year: int, event_name: str, session_type: str):
    """Return a FastF1 Session object (already loaded)."""
    session = fastf1.get_session(year, event_name, session_type)
    session.load()  # fetch telemetry, laps etc.
    return session


@st.cache_data(show_spinner=False)
def get_driver_telemetry(session, driver: str):
    """Return the telemetry DataFrame for a driver in a given session."""
    laps = session.laps.pick_driver(driver)
    # Use the fastest lap (or the first if none marked)
    fastest = laps.pick_fastest()
    return session.get_lap(fastest.lap_number).get_telemetry()


# --------------------------------------------------------------
# 2️⃣ Streamlit layout
# --------------------------------------------------------------
st.set_page_config(page_title="F1 Telemetry & Strategy Studio",
                   layout="wide")

# Sidebar – user inputs
st.sidebar.title("🏁 F1 Telemetry Studio")
year = st.sidebar.selectbox("Season (Year)", options=[2026, 2025, 2024, 2023, 2022])
accent = get_season_accent(year)
inject_css(accent)

# Load calendar once and split
schedule_df = load_event_schedule()
upcoming, completed = split_schedule(schedule_df)

# Upcoming races panel
with st.sidebar.expander("🔜 Upcoming Races", expanded=True):
    for _, row in upcoming.iterrows():
        st.markdown(
            f"**{row['EventName']}** – {row['Date'].strftime('%Y‑%m‑%d %H:%M UTC')} \n"
            f"`⏳ {row['Countdown']}`"
        )

# Completed races – used for dropdown selection
completed_events = completed["EventName"].unique()
event_name = st.sidebar.selectbox("Select Race (Completed)", options=completed_events)
session_type = st.sidebar.selectbox(
    "Session", options=["Q", "R", "S", "FP1", "FP2", "FP3"]
)

# Driver selectors
drivers = [""]  # placeholder, will be filled after session load
if event_name:
    # lazy‑load the session just to retrieve the driver list
    try:
        sess = load_session(year, event_name, session_type)
        drivers = sorted(sess.laps["Driver"].unique().tolist())
    except Exception as e:
        st.error(f"⚠️ Could not load session: {e}")

driver1 = st.sidebar.selectbox("Driver 1", options=drivers, index=0)
driver2 = st.sidebar.selectbox("Driver 2", options=drivers, index=1 if len(drivers) > 1 else 0)

# --------------------------------------------------------------
# 3️⃣ Main visualisation area (tabs)
# --------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Speed / Throttle / Brake", "Track‑Map Heatmap",
     "Tire Strategy", "Mini‑Sector Dominance"]
)

# ------------------------------------------------------------------
# Tab 1 – Speed/Throttle/Brake (Distance axis)
# ------------------------------------------------------------------
with tab1:
    if not driver1 or not driver2:
        st.warning("Select two drivers from the sidebar.")
    else:
        try:
            tele1 = get_driver_telemetry(sess, driver1)
            tele2 = get_driver_telemetry(sess, driver2)

            # Ensure distance column exists (FastF1 provides .Distance)
            for tele in (tele1, tele2):
                if "Distance" not in tele.columns:
                    tele["Distance"] = (tele["X"]**2 + tele["Y"]**2).cumsum() / 1000

            fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            # Speed
            axs[0].plot(tele1["Distance"], tele1["Speed"], label=driver1, color="cyan")
            axs[0].plot(tele2["Distance"], tele2["Speed"], label=driver2, color="magenta")
            axs[0].set_ylabel("Speed (km/h)")
            axs[0].legend()
            # Throttle
            axs[1].plot(tele1["Distance"], tele1["Throttle"], label=driver1, color="cyan")
            axs[1].plot(tele2["Distance"], tele2["Throttle"], label=driver2, color="magenta")
            axs[1].set_ylabel("Throttle (%)")
            # Brake (binary)
            axs[2].plot(tele1["Distance"], tele1["Brake"], label=driver1, color="cyan")
            axs[2].plot(tele2["Distance"], tele2["Brake"], label=driver2, color="magenta")
            axs[2].set_ylabel("Brake (on/off)")
            axs[2].set_xlabel("Distance (km)")
            for ax in axs:
                ax.grid(alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"⚠️ Could not create speed/throttle/brake plot: {e}")

# ------------------------------------------------------------------
# Tab 2 – Track‑Map Speed Heatmap (fastest lap of driver1)
# ------------------------------------------------------------------
with tab2:
    if not driver1:
        st.warning("Select a driver to view the heatmap.")
    else:
        try:
            # Fastest lap telemetry for driver1
            laps = sess.laps.pick_driver(driver1)
            fastest = laps.pick_fastest()
            tel = sess.get_lap(fastest.lap_number).get_telemetry()

            x = tel["X"].values
            y = tel["Y"].values
            speed = tel["Speed"].values

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = mcoll.LineCollection(segments, cmap="plasma",
                                      norm=plt.Normalize(speed.min(), speed.max()))
            lc.set_array(speed)
            lc.set_linewidth(4)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{driver1} – Fastest Lap Speed Heatmap")
            plt.colorbar(lc, label="Speed (km/h)", ax=ax)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"⚠️ Could not create track‑map heatmap: {e}")

# ------------------------------------------------------------------
# Tab 3 – Tire Strategy Gantt chart
# ------------------------------------------------------------------
with tab3:
    try:
        laps_df = sess.laps
        # Determine stint boundaries per driver
        strategy = (
            laps_df
            .groupby(["Driver", "Compound"])
            .agg(StintStart=("LapNumber", "min"),
                 StintEnd=("LapNumber", "max"))
            .reset_index()
        )

        # Map compounds to colours
        comp_color = {
            "Soft": "#FF0000",
            "Medium": "#FFFF00",
            "Hard": "#FFFFFF",
            "Intermediate": "#00FF00",
            "Wet": "#0000FF",
        }

        fig, ax = plt.subplots(figsize=(12, 8))
        drivers_sorted = sorted(strategy["Driver"].unique())
        y_positions = {drv: i for i, drv in enumerate(drivers_sorted)}
        for _, row in strategy.iterrows():
            start = row["StintStart"]
            duration = row["StintEnd"] - start + 1
            ax.broken_barh(
                [(start, duration)],
                (y_positions[row["Driver"]] - 0.4, 0.8),
                facecolors=comp_color.get(row["Compound"], "#888888")
            )
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel("Lap number")
        ax.set_title("Tire Strategy (Stint Gantt)")
        ax.grid(alpha=0.3, axis="x")
        # Legend
        legend_patches = [
            plt.Rectangle((0,0),1,1, color=col, label=comp)
            for comp, col in comp_color.items()
        ]
        ax.legend(handles=legend_patches, title="Compound")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"⚠️ Could not create tire‑strategy chart: {e}")

# ------------------------------------------------------------------
# Tab 4 – Mini‑Sector Dominance
# ------------------------------------------------------------------
with tab4:
    if not driver1 or not driver2:
        st.warning("Select two drivers to compare mini‑sectors.")
    else:
        try:
            # Fastest laps for both drivers
            tel1 = get_driver_telemetry(sess, driver1)
            tel2 = get_driver_telemetry(sess, driver2)

            # Interpolate both telemetry onto equal distance steps
            distance = np.linspace(
                max(tel1["Distance"].min(), tel2["Distance"].min()),
                min(tel1["Distance"].max(), tel2["Distance"].max()),
                500
            )
            speed1 = np.interp(distance, tel1["Distance"], tel1["Speed"])
            speed2 = np.interp(distance, tel2["Distance"], tel2["Speed"])

            # Split into 25 mini‑sectors
            n_sectors = 25
            sector_edges = np.linspace(distance.min(), distance.max(), n_sectors + 1)
            sector_winner = []
            for i in range(n_sectors):
                idx = (distance >= sector_edges[i]) & (distance < sector_edges[i+1])
                avg1 = speed1[idx].mean()
                avg2 = speed2[idx].mean()
                sector_winner.append(driver1 if avg1 > avg2 else driver2)

            # Build coloured track line
            # Use driver1 colour = "cyan", driver2 = "magenta"
            driver_color = {
                driver1: "#00FFFF",
                driver2: "#FF00FF"
            }

            # Use driver1's XY for geometry (any driver works for layout)
            x = tel1["X"].values
            y = tel1["Y"].values
            # Create segments and assign colour per mini‑sector
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Map each segment to a sector index
            seg_dist = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=2)).flatten()
            cum_dist = np.cumsum(seg_dist)
            cum_dist = np.insert(cum_dist, 0, 0)

            seg_colors = []
            for cd in cum_dist:
                # Find sector
                sector_idx = np.searchsorted(sector_edges, cd, side="right") - 1
                sector_idx = min(sector_idx, n_sectors-1)
                winner = sector_winner[sector_idx]
                seg_colors.append(driver_color[winner])

            lc = mcoll.LineCollection(segments, colors=seg_colors, linewidths=4)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title("Mini‑Sector Dominance")
            # Legend
            legend_handles = [
                plt.Line2D([0], [0], color=driver_color[driver1], lw=4, label=driver1),
                plt.Line2D([0], [0], color=driver_color[driver2], lw=4, label=driver2)
            ]
            ax.legend(handles=legend_handles, title="Faster driver per sector")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"⚠️ Could not create mini‑sector plot: {e}")