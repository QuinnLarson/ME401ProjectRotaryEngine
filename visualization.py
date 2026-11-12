"""Visualization and structured data export pipeline for the ME401 rotary simulation.

Exports:
- CSV tables (`speed_sweep_results.csv`, `parametric_cr_recess.csv`,
  `parametric_h2_blend.csv`, `optimal_cycle_trajectory.csv`)
- JSON summary (`simulation_summary.json`)
- High-resolution comparative figures (PNG + PDF) in `Project Code/figures/`:
  1. Rotary vs. Piston Baseline (Efficiency, BSFC, and Power across RPM)
  2. Breakdown of Loss Mechanisms across Speeds (Heat Transfer, Seal Leakage, Friction, Pumping)
  3. Optimal Operating Condition P-V and T-theta Cycle Diagrams
  4. Parametric Sensitivity (CR & LDR/SMR Recess, and Hydrogen Mass Fraction x_H2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SimulationConfig
from optimizer import OptimizationSummary


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rotary_vs_piston(
    merged_df: pd.DataFrame,
    opt_rpm: float,
    target_kw: float,
    figures_dir: Path,
) -> None:
    """Figure 1: Rotary vs. Reciprocating Piston Baseline across RPM."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    rpms = merged_df["rpm"]
    rot_feas = merged_df["feasible"]
    pis_feas = merged_df["piston_feasible"]

    # Panel (a): Brake Thermal Efficiency & System Efficiency
    ax = axes[0]
    ax.plot(rpms, merged_df["eta_bth"] * 100.0, "o-", color="#1f77b4", lw=2, label="Wankel BTE")
    ax.plot(rpms, merged_df["eta_system"] * 100.0, "s--", color="#1f77b4", alpha=0.65, lw=1.5, label="Wankel Sys Eff (BTE*eta_gen)")
    ax.plot(rpms, merged_df["piston_eta_bth"] * 100.0, "^-", color="#d62728", lw=2, label="Atkinson Piston BTE")
    ax.plot(rpms, merged_df["piston_eta_system"] * 100.0, "v--", color="#d62728", alpha=0.65, lw=1.5, label="Atkinson Sys Eff")
    ax.axvline(opt_rpm, color="#2ca02c", ls=":", lw=1.8, label=f"Rotary N_opt = {opt_rpm:.0f} RPM")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Efficiency [%]")
    ax.set_title("(a) Thermal & System Efficiency")
    ax.legend(loc="lower right", frameon=True)

    # Panel (b): BSFC Comparison
    ax = axes[1]
    ax.plot(rpms, merged_df["bsfc_g_kwh"], "o-", color="#1f77b4", lw=2, label="Wankel (H2-Gasoline)")
    ax.plot(rpms, merged_df["piston_bsfc_g_kwh"], "^-", color="#d62728", lw=2, label="Atkinson Piston (Gasoline)")
    opt_row = merged_df[merged_df["rpm"] == opt_rpm].iloc[0]
    ax.scatter([opt_rpm], [opt_row["bsfc_g_kwh"]], color="#2ca02c", s=90, zorder=5, label=f"Optimal: {opt_row['bsfc_g_kwh']:.1f} g/kWh")
    ax.axvline(opt_rpm, color="#2ca02c", ls=":", lw=1.5)
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("BSFC [g / (kW·h)]")
    ax.set_title("(b) Brake Specific Fuel Consumption")
    ax.legend(loc="upper right", frameon=True)

    # Panel (c): WOT Capability vs. Matched Electrical Output
    ax = axes[2]
    ax.plot(rpms, merged_df["p_elec_wot_kw"], "o-", color="#1f77b4", lw=2, label="Wankel WOT Elec. Capability")
    ax.plot(rpms[rot_feas], merged_df.loc[rot_feas, "p_elec_kw"], "s-", color="#2ca02c", lw=2, label="Wankel Load-Matched Output")
    ax.plot(rpms, merged_df["piston_p_elec_wot_kw"], "^--", color="#d62728", lw=1.8, label="Piston WOT Elec. Capability")
    ax.axhline(target_kw, color="#111111", ls="-.", lw=1.4, label=f"Target Grid Load ({target_kw:.1f} kW)")
    if pis_feas.any():
        ax.axvspan(rpms[rot_feas].min(), rpms[rot_feas].max(), color="#2ca02c", alpha=0.07, label="Wankel Feasible Window")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Electrical Power [kW]")
    ax.set_title("(c) Generator Power & Feasibility")
    ax.legend(loc="upper left", frameon=True)

    fig.suptitle("Wankel Hybrid Range Extender vs. Equivalent Atkinson Piston Baseline", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig1_rotary_vs_piston_performance")


def plot_loss_breakdown(
    sweep_df: pd.DataFrame,
    opt_rpm: float,
    figures_dir: Path,
) -> None:
    """Figure 2: Opposing Loss Mechanisms Across Engine Speeds."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    rpms = sweep_df["rpm"]

    # Panel (a): Energy Loss Fractions (% of fuel input energy)
    ax = axes[0]
    ax.plot(rpms, sweep_df["q_wall_loss_pct"], "o-", color="#e6550d", lw=2, label="Convective Wall Heat Loss")
    ax.plot(rpms, sweep_df["h_leak_loss_pct"], "s-", color="#756bb1", lw=2, label="Seal Leakage Energy Loss")
    ax.plot(rpms, sweep_df["friction_loss_pct"], "^-", color="#31a354", lw=2, label="Mechanical Friction (Rubbing + Seal Drag)")
    ax.plot(rpms, sweep_df["pumping_loss_pct"], "d-", color="#636363", lw=1.8, label="Intake/Exhaust Pumping Loss")
    ax.axvline(opt_rpm, color="#2ca02c", ls=":", lw=1.8, label=f"N_opt = {opt_rpm:.0f} RPM")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Fraction of Fuel Chemical Energy [%]")
    ax.set_title("(a) Cycle Energy Loss Mechanisms vs. RPM")
    ax.legend(loc="upper right", frameon=True)

    # Panel (b): Physical Trade-Off Drivers (Blow-by Mass % vs FMEP & Rubbing/Seal Breakdown)
    ax = axes[1]
    ax2 = ax.twinx()
    l1 = ax.plot(rpms, sweep_df["leak_mass_pct"], "s-", color="#756bb1", lw=2, label="Trapped Mass Blow-By [%] (left)")
    l2 = ax2.plot(rpms, sweep_df["fmep_rubbing_bar"], "o--", color="#3182bd", lw=1.8, label="Bearing/Gear Rubbing FMEP [bar] (right)")
    l3 = ax2.plot(rpms, sweep_df["fmep_seal_drag_bar"], "^-", color="#de2d26", lw=1.8, label="Apex/Side Seal Drag FMEP [bar] (right)")
    ax.axvline(opt_rpm, color="#2ca02c", ls=":", lw=1.8)
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Closed-Cycle Mass Blow-By [%]", color="#756bb1")
    ax2.set_ylabel("Friction Mean Effective Pressure [bar]", color="#333333")
    ax.set_title("(b) Low-RPM Blow-By vs. High-RPM Friction Trade-Off")
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="upper center", frameon=True)

    fig.suptitle("Rotary Loss Decomposition Across Speed Sweep (2,000 - 7,000 RPM)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig2_loss_mechanisms_vs_rpm")


def plot_optimal_cycle_diagrams(
    opt_summary: OptimizationSummary,
    figures_dir: Path,
) -> None:
    """Figure 3: P-V and T-theta Cycle Diagrams at Optimal Operating Speed."""
    _configure_plot_style()
    cyc = opt_summary.optimal_cycle
    pt = opt_summary.optimal_point

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    v_cm3 = cyc.volume_m3 * 1e6
    p_bar = cyc.pressure_pa / 1e5
    th = cyc.theta_deg

    # Panel (a): Closed P-V Indicator Diagram
    ax = axes[0]
    mask_in = th <= 270.0
    mask_comp = (th >= 270.0) & (th <= 540.0)
    mask_exp = (th >= 540.0) & (th <= 810.0)
    mask_exh = th >= 810.0

    ax.plot(v_cm3[mask_in], p_bar[mask_in], color="#1f77b4", lw=1.8, label="Intake (0-270°)")
    ax.plot(v_cm3[mask_comp], p_bar[mask_comp], color="#ff7f0e", lw=2.0, label="Compression (270-540°)")
    ax.plot(v_cm3[mask_exp], p_bar[mask_exp], color="#d62728", lw=2.2, label="Combustion & Expansion (540-810°)")
    ax.plot(v_cm3[mask_exh], p_bar[mask_exh], color="#7f7f7f", lw=1.6, ls="--", label="Exhaust (810-1080°)")

    idx_spk = int(np.argmin(np.abs(th - cyc.combustion_params.spark_timing_deg)))
    ax.scatter([v_cm3[idx_spk]], [p_bar[idx_spk]], color="#000000", s=50, zorder=6, label=f"Spark ({cyc.combustion_params.spark_timing_deg:.0f}° EA)")
    ax.set_xlabel("Chamber Volume V [cm³]")
    ax.set_ylabel("In-Chamber Pressure P [bar]")
    ax.set_title(f"(a) Closed P-V Diagram @ {pt.rpm:.0f} RPM (P_max={pt.p_max_bar:.1f} bar, BTE={pt.eta_bth*100:.1f}%)")
    ax.legend(loc="upper right", frameon=True)

    # Panel (b): T-theta Trajectory across 1080 deg
    ax = axes[1]
    ax.plot(th, cyc.temperature_k, color="#d62728", lw=2.2, label="Gas Temperature T(θ)")
    for bnd, lbl in [(270, "BDC Intake"), (540, "TDC Comb."), (810, "BDC Exp.")]:
        ax.axvline(bnd, color="#555555", ls=":", lw=1.2)
        ax.text(bnd + 8, 0.04, lbl, transform=ax.get_xaxis_transform(), rotation=90, fontsize=8, color="#444444")
    ax.axvline(cyc.combustion_params.spark_timing_deg, color="#1f77b4", ls="--", lw=1.4, label=f"Spark ({cyc.combustion_params.spark_timing_deg:.0f}°)")
    ax.axhline(430.0, color="#8c564b", ls="-.", lw=1.2, label="Wall Temp (430 K)")
    ax.set_xlim(0, 1080)
    ax.set_xlabel("Eccentric Shaft Angle θ [deg]")
    ax.set_ylabel("In-Chamber Gas Temperature T [K]")
    ax.set_title(f"(b) T-θ Trajectory Over 1080° Cycle (T_max={pt.t_max_k:.0f} K)")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle(f"Optimal Range-Extender Operating Cycle (N_opt = {pt.rpm:.0f} RPM, P_elec = {pt.p_elec_kw:.2f} kW)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig3_optimal_cycle_diagrams")


def plot_parametric_sweeps(
    cr_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Figure 4: Compression Ratio / Recess & Hydrogen Doping Sensitivity."""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # Panel (a): CR & Recess Comparison
    ax = axes[0]
    styles = {
        ("LDR", 8.33): ("o--", "#1f77b4"),
        ("LDR", 9.55): ("o-", "#1f77b4"),
        ("LDR", 10.18): ("o-.", "#084594"),
        ("SMR", 8.33): ("^--", "#e6550d"),
        ("SMR", 9.55): ("^-", "#e6550d"),
        ("SMR", 10.18): ("^-.", "#a63603"),
    }
    for (rec, cr), grp in cr_df.groupby(["recess_type", "CR"]):
        ls, col = styles.get((rec, cr), ("o-", "#333333"))
        ax.plot(grp["rpm"], grp["eta_bth_pct"], ls, color=col, lw=1.7, label=f"{rec}, CR={cr:.2f}")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Brake Thermal Efficiency [%]")
    ax.set_title("(a) Recess Pocket (LDR vs. SMR) & CR Sensitivity")
    ax.legend(ncol=2, loc="lower center", frameon=True)

    # Panel (b): Hydrogen Blending Impact on Efficiency and CO2
    ax = axes[1]
    ax2 = ax.twinx()
    colors_h2 = {0.0: "#4d4d4d", 0.1: "#2b8cbe", 0.2: "#045a8d", 0.3: "#2ca02c"}
    for x_h2, grp in h2_df.groupby("x_H2"):
        col = colors_h2.get(float(x_h2), "#333333")
        ax.plot(grp["rpm"], grp["eta_bth_pct"], "o-", color=col, lw=1.8, label=f"BTE: x_H2={x_h2:.1f}")
        ax2.plot(grp["rpm"], grp["co2_g_kwh"], "x:", color=col, lw=1.4, label=f"CO2: x_H2={x_h2:.1f}")
    ax.set_xlabel("Engine Speed [RPM]")
    ax.set_ylabel("Brake Thermal Efficiency [%]")
    ax2.set_ylabel("Specific CO2 Emission [g / (kW·h)]")
    ax.set_title("(b) Hydrogen Mass Fraction (x_H2) Effect on BTE & CO2")
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    fig.suptitle("Parametric Design & Fuel Blending Sensitivity", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig4_parametric_cr_and_h2")


def export_all_outputs(
    sim_cfg: SimulationConfig,
    opt_summary: OptimizationSummary,
    piston_df: pd.DataFrame,
    cr_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    base_dir: Path,
) -> Dict[str, Path]:
    """Export all CSV/JSON datasets and high-resolution figures."""
    output_dir = base_dir / "output"
    figures_dir = base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.merge(opt_summary.sweep_df, piston_df, on="rpm", how="left")
    merged_df["eta_bth_delta_pct_pts"] = (merged_df["eta_bth"] - merged_df["piston_eta_bth"]) * 100.0
    merged_df["bsfc_delta_g_kwh"] = merged_df["bsfc_g_kwh"] - merged_df["piston_bsfc_g_kwh"]

    sweep_csv = output_dir / "speed_sweep_results.csv"
    merged_df.to_csv(sweep_csv, index=False, float_format="%.6g")

    cr_csv = output_dir / "parametric_cr_recess.csv"
    cr_df.to_csv(cr_csv, index=False, float_format="%.6g")

    h2_csv = output_dir / "parametric_h2_blend.csv"
    h2_df.to_csv(h2_csv, index=False, float_format="%.6g")

    cyc = opt_summary.optimal_cycle
    traj_df = pd.DataFrame(
        {
            "theta_deg": cyc.theta_deg,
            "volume_cm3": cyc.volume_m3 * 1e6,
            "pressure_bar": cyc.pressure_pa / 1e5,
            "temperature_k": cyc.temperature_k,
            "mass_mg": cyc.mass_kg * 1e6,
            "surface_area_cm2": cyc.area_m2 * 1e4,
            "burned_fraction": cyc.burned_fraction,
            "dq_comb_j_per_deg": cyc.dq_comb_ddeg,
            "dq_wall_j_per_deg": cyc.dq_wall_ddeg,
            "dm_leak_mg_per_deg": cyc.dm_leak_ddeg * 1e6,
        }
    )
    traj_csv = output_dir / "optimal_cycle_trajectory.csv"
    traj_df.to_csv(traj_csv, index=False, float_format="%.6g")

    opt_pt = opt_summary.optimal_point
    pis_at_opt = piston_df[piston_df["rpm"] == opt_pt.rpm].iloc[0]
    pis_feas_df = piston_df[piston_df["piston_feasible"]]
    if not pis_feas_df.empty:
        pis_best = pis_feas_df.sort_values("piston_eta_system", ascending=False).iloc[0]
    else:
        pis_best = piston_df.sort_values("piston_p_elec_wot_kw", ascending=False).iloc[0]

    summary_payload = {
        "configuration": {
            "rotor_radius_m": sim_cfg.geometry.rotor_radius,
            "eccentricity_m": sim_cfg.geometry.eccentricity,
            "rotor_width_m": sim_cfg.geometry.rotor_width,
            "swept_volume_cm3": sim_cfg.geometry.swept_volume * 1e6,
            "compression_ratio": sim_cfg.geometry.compression_ratio,
            "recess_type": sim_cfg.geometry.recess_type,
            "h2_mass_fraction": sim_cfg.fuel.h2_mass_fraction,
            "excess_air_ratio_lambda": sim_cfg.fuel.excess_air_ratio,
            "composite_lhv_mj_per_kg": sim_cfg.fuel.composite_lhv / 1e6,
            "target_electric_power_kw": sim_cfg.hybrid.target_electric_power_kw,
        },
        "optimal_rotary_point": {
            "optimal_rpm": opt_pt.rpm,
            "p_elec_kw": opt_pt.p_elec_kw,
            "p_brake_kw": opt_pt.p_brake_kw,
            "eta_ith_pct": opt_pt.eta_ith * 100.0,
            "eta_bth_pct": opt_pt.eta_bth * 100.0,
            "eta_system_pct": opt_pt.eta_system * 100.0,
            "bsfc_g_kwh": opt_pt.bsfc_g_kwh,
            "leak_mass_pct": opt_pt.leak_mass_pct,
            "q_wall_loss_pct": opt_pt.q_wall_loss_pct,
            "friction_loss_pct": opt_pt.friction_loss_pct,
            "energy_balance_rel_error": opt_pt.energy_balance_rel_error,
        },
        "piston_baseline_comparison": {
            "piston_at_rotary_opt_rpm": {
                "rpm": float(pis_at_opt["rpm"]),
                "eta_bth_pct": float(pis_at_opt["piston_eta_bth"] * 100.0),
                "eta_system_pct": float(pis_at_opt["piston_eta_system"] * 100.0),
                "bsfc_g_kwh": float(pis_at_opt["piston_bsfc_g_kwh"]),
            },
            "piston_global_optimal": {
                "rpm": float(pis_best["rpm"]),
                "eta_bth_pct": float(pis_best["piston_eta_bth"] * 100.0),
                "eta_system_pct": float(pis_best["piston_eta_system"] * 100.0),
                "bsfc_g_kwh": float(pis_best["piston_bsfc_g_kwh"]),
            },
        },
    }
    summary_json = output_dir / "simulation_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    plot_rotary_vs_piston(merged_df, opt_pt.rpm, sim_cfg.hybrid.target_electric_power_kw, figures_dir)
    plot_loss_breakdown(opt_summary.sweep_df, opt_pt.rpm, figures_dir)
    plot_optimal_cycle_diagrams(opt_summary, figures_dir)
    plot_parametric_sweeps(cr_df, h2_df, figures_dir)

    return {
        "speed_sweep_csv": sweep_csv,
        "parametric_cr_csv": cr_csv,
        "parametric_h2_csv": h2_csv,
        "trajectory_csv": traj_csv,
        "summary_json": summary_json,
        "figures_dir": figures_dir,
    }
