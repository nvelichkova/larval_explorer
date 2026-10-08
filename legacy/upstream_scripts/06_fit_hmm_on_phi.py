#!/usr/bin/env python3
"""06_fit_hmm_on_phi.py
=================================================
Fit a 3-state Gaussian HMM using only the phi variable from the RDP step table.

The model is used as a geometric coarse-graining device: it partitions step
sequences into internally coherent geometric regimes based on whether turning
direction tends to persist or reverse. This script standardizes phi, fits the
model, decodes the most likely state sequence, and saves both the state labels
and the state posterior probabilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


def fit_hmm_on_phi(input_csv: Path, output_csv: Path, figure_dir: Path, n_states: int = 3) -> pd.DataFrame:
    """Fit a Gaussian HMM on phi and save decoded states + posteriors."""
    steps = pd.read_csv(input_csv).dropna(subset=["phi", "animal ID"]).copy()

    features = ["phi"]
    scaler = StandardScaler()
    standardized = steps.copy()
    standardized[features] = scaler.fit_transform(steps[features])

    sequences = []
    lengths = []
    for _, group in standardized.groupby("animal ID"):
        sequences.append(group[features].values)
        lengths.append(len(group))
    X_all = np.vstack(sequences)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
        init_params="",
    )

    # Manual initialization reproduces the original modeling choice.
    mu_phi = 0.8
    sigma_phi_fast = 0.15
    sigma_phi_mixed = 1.00
    mu = scaler.mean_[0]
    sigma = scaler.scale_[0]
    to_z = lambda value: (value - mu) / sigma
    to_z_variance = lambda variance: variance / (sigma ** 2)

    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.ones((n_states, n_states)) / n_states
    model.means_ = np.array([[to_z(+mu_phi)], [to_z(-mu_phi)], [to_z(0.0)]])
    model.covars_ = np.array([
        [to_z_variance(sigma_phi_fast)],
        [to_z_variance(sigma_phi_fast)],
        [to_z_variance(sigma_phi_mixed)],
    ])

    model.fit(X_all, lengths)
    states = model.predict(X_all, lengths)
    probabilities = model.predict_proba(X_all)

    steps["estado"] = states
    steps[["p_est0", "p_est1", "p_est2"]] = probabilities

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    steps.to_csv(output_csv, index=False)
    print(f"Saved HMM-labelled step table to: {output_csv}")

    # Optional diagnostic plots: state-wise and overlaid phi distributions.
    figure_dir.mkdir(parents=True, exist_ok=True)
    palette = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(11, 4))
    for state_id, axis, color in zip(range(n_states), axes, palette):
        sns.kdeplot(steps.loc[steps["estado"] == state_id, "phi"], ax=axis, bw_adjust=0.7, fill=False, color=color)
        axis.set_title(f"State {state_id}")
    plt.suptitle("State-wise distributions of $\phi$")
    plt.tight_layout()
    fig.savefig(figure_dir / "phi_distribution_by_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.5, 4.5))
    for state_id, color in zip(range(n_states), palette):
        sns.kdeplot(steps.loc[steps["estado"] == state_id, "phi"], ax=axis, bw_adjust=0.7, fill=False, color=color, label=f"State {state_id}")
    axis.set_title("Overlaid distributions of $\phi$")
    axis.legend(frameon=False, bbox_to_anchor=(1.05, 1.0))
    fig.savefig(figure_dir / "phi_distribution_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return steps


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "outputs" / "05_rdp_steps" / "rdp_steps.csv"
    output_csv = project_root / "outputs" / "06_hmm" / "rdp_steps_with_hmm_states.csv"
    figure_dir = project_root / "outputs" / "06_hmm" / "figures"
    fit_hmm_on_phi(input_csv, output_csv, figure_dir)


if __name__ == "__main__":
    main()
