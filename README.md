# 🏎️ F1 Telemetry & Strategy Studio

A full-stack Formula 1 telemetry and race strategy visualization studio built with **Streamlit**, **FastF1**, **Pandas**, and **Matplotlib**.

---

## ✨ Features

1. **📈 Speed, Throttle & Brake Comparison:** High-frequency, multi-axis telemetry traces comparing two drivers' fastest laps over track distance.
2. **🗺️ Track Speed Heatmap:** Circuit layout mapped with Matplotlib `LineCollection`, gradient-colored by car speed using the `plasma` colormap.
3. **🔧 Tire Strategy Chart:** Horizontal stint Gantt chart for the entire grid with Pirelli compound colors (Soft 🔴, Medium 🟡, Hard ⚪, Intermediate 🟢, Wet 🔵).
4. **🔬 Mini-Sector Dominance:** Track divided into 25 mini-sectors, colored dynamically by each driver's team color based on sector speed dominance.
5. **🗓️ Automated Season Calendar:** Dynamically splits rounds into upcoming (with countdowns) and completed races.
6. **🎨 Season-Adaptive Glassmorphic UI:** Dark mode styling with dynamic accent themes (Cyberpunk Cyan for 2026, Classic F1 Red for ≤2025).

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/f1-telemetry-studio.git
cd f1-telemetry-studio
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Formula 1 credentials
FastF1 requires a free [Formula 1 account](https://account.formula1.com/) for live-timing telemetry. Create `.streamlit/secrets.toml`:
```toml
F1_USERNAME = "your_email@example.com"
F1_PASSWORD = "your_password"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repository to GitHub.
2. In your [Streamlit Cloud Dashboard](https://share.streamlit.io/), click **New app** and select your repository.
3. In **Advanced Settings**, select **Python 3.11** or **Python 3.12**.
4. In **Settings → Secrets**, paste your F1 credentials:
   ```toml
   F1_USERNAME = "your_email@example.com"
   F1_PASSWORD = "your_password"
   ```
5. Click **Deploy**!

---

## 🛠️ Tech Stack
* **Python 3.11 / 3.12 / 3.13**
* **Streamlit** (UI & Reactive State)
* **FastF1** (F1 Live Timing API)
* **Pandas & NumPy** (Vectorized Telemetry Processing)
* **Matplotlib** (Custom Collections & Visualizations)
