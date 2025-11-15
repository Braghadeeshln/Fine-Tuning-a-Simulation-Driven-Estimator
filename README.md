# Fine-Tuning-a-Simulation-Driven-Estimator

## Implementations inside the notebooks

- **TS-pre:** Pretrain a multi-head MLP on a large offline simulation bank.
- **OOD Gate:** OOD detector ($\chi^2$ Wald test) in whitened feature space around the initial TS-pre parameter estimate.
- **GN update:** Trust-region line-searched GN to refine TS-pre parameter estimate before generating the synthetic dataset for retraining heads.
- **Synthetic dataset and retrain:** Freeze trunk, fine-tune only heads with Huber loss and output-dimension weights using synthetic dataset generated via local Fisher $\mathrm{diag}(G)$ and sensitivity-based sampling.
  - We approximate the inverse of $G$ via the SVD of $J$: keep the top-$r$ right singular vectors $V_r$ and project steps as $V_r V_r^\top \Delta\theta$. This behaves like a low-rank inverse $G^{-1} \approx V_r \Sigma_r^{-2} V_r^\top$.
- **Baselines:**  
  - **PEM**: trajectory least-squares with GN + backtracking.
    - For cascaded water tanks, PEM updates are constrainted to a box $\Theta=[\Theta_{lo}, \Theta_{hi}]$ that contains the pretraining set $\Theta_p$.
  - **Dual-EKF**
- **Determinism:** fixed seeds; deterministic torch; TF32 off; cuDNN deterministic.

## Environment

Use any recent Python + PyTorch (GPU recommended).

# Scientific libraries
pip install numpy pandas matplotlib scikit-learn

## Quick Start

1. **Open a notebook (preferably in Colab)**
   - `last_layer_tanks_LCSS_final.ipynb` (Cascaded Water Tanks), or  
   - `last_layer_VDP_LCSS_final.ipynb` (Van der Pol).

2. **Run cells top-to-bottom.** The notebook will:
   - Generate the **offline bank** for pretraining.
     - Simulator state clipping (VDP & Tanks): We clip simulator states each step for stability, and this applies to all methods (TS-pre, TS-fine, PEM, Dual-EKF) since they         all use the same data-generating simulator.
   - **Pretrain** the MLP head (**TS-pre**).
   - Run **Monte-Carlo (MC)** evaluations for each OOD scenario:
     - **Gate** (OOD detection in whitened feature space)  
     - **GN** warm-start (trust-region Gauss–Newton)  
     - **TS-fine** (Fisher+sensitivity sampling followed by last-layer fine-tuning)  
     - **baselines** (PEM, Dual-EKF)

3. **Artifacts** are saved to `results_tanks/` or `results_vdp/`:
   - `mse_boxplot_<plant>_<scenario>_noisy.pdf`
   - `<plant>_<scenario>_mc<N>.npz` — contains estimates, MSEs, times, and plotting arrays.
