"""Five physical-unit profiles from a deterministically selected training run."""
from __future__ import annotations

from .runtime import RESULTS, write_json
from .data_io import development_pool, manifest, load_runs, UNITS

import matplotlib.pyplot as plt


def main() -> None:
    pool = development_pool()
    if pool:
        key = pool[0]
        rule = "Minimum (source_dataset, integer run_id) in D1-train intersection D5-train; no appearance-based selection"
    else:
        key = sorted(manifest("D1")["train"])[0]
        rule = "Common pool empty: minimum (source_dataset, integer run_id) in D1-train"
    run = load_runs([key])
    v_a, mdot, p_b, b_hat = run.inputs[0]
    labels = [r"$n_e$ [m$^{-3}$]", r"$T_e$ [eV]", "Potential [V]",
              r"$u_i$ (Xe$^+$) [m/s]", r"$n_n$ [m$^{-3}$]"]
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 11), sharex=True, facecolor="white")
    for j, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(run.z * 1e3, run.outputs[0, j], color="#17618c", linewidth=1.8)
        ax.axvline(25, color="#666666", linestyle="--", linewidth=1,
                   label="Channel exit (25 mm)")
        ax.set_ylabel(label, fontsize=11)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 80)
        ax.tick_params(labelsize=10)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("Axial position [mm]", fontsize=11)
    fig.suptitle(f"One simulation · run {key.run_id}\n{key.source_dataset}\n"
                 f"$V_a$ = {v_a:.3f} V    mass flow = {mdot * 1e6:.4f} mg/s    "
                 f"$P_b$ = {p_b * 1e6:.3f} µTorr    $\\hat B$ = {b_hat:.5f}", fontsize=12, y=0.99)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(RESULTS / "example_run.png", dpi=180, facecolor="white")
    plt.close(fig)
    write_json(RESULTS / "example_run_metadata.json", {**key.as_dict(), "selection_rule": rule,
        "in_D1_and_D5_training": key in set(pool), "inputs": dict(zip(("V_a", "mdot", "P_b", "B_hat"), map(float, run.inputs[0]))),
        "units": UNITS, "fields": ["ne", "Tev", "potential", "ui[0]", "nn"],
        "stored_ui_shape": [3, 202], "plotted_profile_shape": [5, 202],
        "axial_unit_in_figure": "mm", "channel_exit_mm": 25,
        "figure": "results/stage1/example_run.png"})
    print(f"Saved five-panel example: {key}")


if __name__ == "__main__":
    main()
