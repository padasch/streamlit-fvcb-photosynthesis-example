# FvCB Webapp

This is a Streamlit app for interactively exploring a simplified Farquhar, von Caemmerer & Berry (FvCB)-style
model response surface:

- `A_net` vs **Light (PAR)**
- `A_net` vs **C_i**
- `A_net` vs **T_leaf**
- `A_net` vs **VPD**

You can:

- Change default settings with sliders.
- Add comparison curves by capturing the current defaults as a named snapshot.
- Overlay many curves in one chart to compare how alternative defaults shift the response.
- Fix the X-axis and Y-axis explicitly, or keep each one dynamic (auto-scaled).
- Toggle decomposition lines to show `A_c` and `A_j` background curves and identify the limiting term.
- Plotting now uses Altair in Streamlit with explicit color labels for A_c/A_j limiting regimes.

## Run

```bash
# from your project folder
bash run.sh
```

If you prefer manual steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Public deployment

This repo is ready for Streamlit Community Cloud deployment.

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Choose **New app** and connect your repo.
4. Select the branch and set the entry file to `app.py`.
5. Ensure the package install command is `pip install -r requirements.txt`.
6. Click **Deploy** and share the generated URL.

### Alternative: Render

- Add a web service on Render.
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Set it to run from the repo root.

## Recommended deployment files

This project includes:

- `requirements.txt` (dependencies)
- `.streamlit/config.toml` (headless + host config)
- `run.sh` (local run helper)

## Notes

- This is an educational/interactive implementation for quick exploration.
- The VPD effect is modeled as a tunable stress scalar and is not a fully mechanistic stomatal conductance model.
