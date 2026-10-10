# Where to put the notebook outputs

Copy the three folders the notebook wrote (Google Drive: `MyDrive/pinn_project/`) into this folder:

    data/results/       results_quick_gpu.json (or .csv), burgers_reference_N2048.npz, findings_*.txt
    data/checkpoints/   *.pt   (e.g. burgers_adam_lbfgs_s1234__quick_gpu.pt)
    data/figures/       *.png  (optional, shown on the Results page)

Notes
- Use the files from ONE run (same RUN_TAG suffix), or keep several: the app lets you pick the run.
- Checkpoints are small (tens to hundreds of KB). If you do not want to commit them, upload them in the app sidebar.
- Checkpoints are pickle-based: only use files you created yourself.
