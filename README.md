# PINN vs. classical solvers: Streamlit dashboard

Interactive front-end for `pinn_project.ipynb`.

## Layout

    app.py                      home page (headline comparison, data status)
    pages/1_Results.py          results table, accuracy-vs-cost plots, notebook figures
    pages/2_Classical_Solvers.py  live FD / Crank-Nicolson / spectral solvers
    pages/3_PINN_Explorer.py    load trained checkpoints, compare with exact/reference solutions
    pages/4_Inverse_and_Ablation.py
    pinn_lib/                   code extracted from the notebook (PDEs, solvers, metrics, plots, model loader)
    data/                       PUT YOUR NOTEBOOK OUTPUTS HERE (see data/README.md)
    tests/smoke_test.py         headless check of the numerical code

## 1. Add your outputs

Copy from Google Drive `pinn_project/` into `data/`:
`results/` (results_*.json or .csv, burgers_reference_N2048.npz), `checkpoints/*.pt`, `figures/*.png`.
The app also works with nothing in `data/`: you can upload the results file (and a checkpoint) in the sidebar.

## 2. Run locally (optional)

    pip install -r requirements.txt
    streamlit run app.py



