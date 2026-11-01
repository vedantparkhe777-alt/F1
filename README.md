# F1 Telemetry & Strategy Studio

A single‑file Streamlit app that lets you explore Formula 1 telemetry, strategy and visualisations using **FastF1**, **Pandas** and **Matplotlib**.

## Features

- Dark, glass‑morphic UI that automatically colours the accent based on the selected season (2025 = classic Red, 2026 = Cyberpunk Cyan).
- Sidebar controls for season, race, session and driver selection.
- Automatic calendar loading with upcoming‑race countdown and completed‑race dropdown.
- Four core visualisations:
  1. Speed / Throttle / Brake traces (distance axis).
  2. Track‑map speed heatmap of the fastest lap.
  3. Tire‑strategy Gantt chart.
  4. Mini‑sector dominance map.
- FastF1 cache (`f1_cache`) and Streamlit `@st.cache_data` to minimise data fetching.
- Graceful error handling with `st.error`.

## Quick start (local)

```powershell
# 1️⃣ Clone / copy the repository
git clone <YOUR_GITHUB_URL>
cd F1

# 2️⃣ Create a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or   .venv\Scripts\activate.bat   # cmd.exe

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud (or any Streamlit‑compatible platform)

1. **Push the code to GitHub**
   ```bash
   git add app.py README.md requirements.txt
   git commit -m "Initial commit – F1 Telemetry Studio"
   git push origin main   # or the branch you prefer
   ```
2. **Create a new app on Streamlit Cloud**
   - Sign in at https://share.streamlit.io.
   - Click **New app**, select the repository, branch and set *Main file path* to `app.py`.
   - In *Advanced settings* add the environment variable `FASTF1_CACHE` if you want a persistent cache (optional).
   - Click **Deploy**.

   Streamlit will automatically install the packages listed in `requirements.txt` and launch the app.

3. **Custom domain (optional)** – after the app is deployed, you can map a custom domain via the Streamlit Cloud UI.

## Repository layout

```
├─ app.py          # Streamlit application (single file)
├─ README.md       # This file
└─ requirements.txt
```

## Required Python packages

The `requirements.txt` file contains all dependencies:

```
streamlit>=1.38.0
fastf1>=3.5.0
pandas>=2.2.0
matplotlib>=3.9.0
numpy>=2.0.0
```

> **Note** – FastF1 will download the FIA telemetry files on‑the‑fly and cache them under the `f1_cache` folder created in the working directory.

## License & attribution

- This project uses the open‑source **FastF1** library (MIT licence) for data retrieval.
- Streamlit, Pandas and Matplotlib are also open source.
- Feel free to fork, modify and share!

---

*Enjoy analysing the races and discovering strategic insights!*