"""Main execution entry point for the ME401 Rotary Hybrid Range-Extender Simulation.

Pipeline:
1. Geometry & Kinematics Input Display (interactive or batch defaults for R, e, b, CR).
2. Physics & Numerical Sanity Checks (volume conservation, analytical derivative match,
   First Law closed-cycle energy balance residual).
3. Speed Sweep Optimization (2,000 - 7,000 RPM in 250-RPM increments) coupled to
   generator efficiency curve and 15 kW electrical demand.
4. Equivalent Standard Gasoline Reciprocating Atkinson-Cycle Benchmark Comparison.
5. Parametric Sensitivity Sweeps (CR = 8.33/9.55/10.18, LDR vs. SMR, x_H2 = 0.0-0.30).
6. Tabular CSV/JSON Export & Publication-Quality Comparative Plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Callable, TypeVar

import numpy as np

from benchmark import run_piston_benchmark_sweep
from config import FuelBlendConfig, HybridGeneratorConfig, OperatingConditions, RotaryGeometry, SimulationConfig
from cycle_solver import WankelCycleSolver
from kinematics import WankelKinematicsLookup, chamber_volume, display_and_refine_geometry, volume_derivative_deg
from optimizer import run_parametric_sensitivity, run_speed_sweep_optimization
from visualization import export_all_outputs

_T = TypeVar("_T")


def _resolve_output_dir() -> Path:
    """Return writable base directory next to main.py (or executable if frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _prompt_with_default(
    prompt_label: str,
    default: _T,
    parser: Callable[[str], _T],
    validator: Callable[[_T], bool],
    error_msg: str,
    fmt: str = "",
) -> _T:
    """Prompt until a valid value is entered; empty input keeps default."""
    default_str = format(default, fmt) if fmt else str(default)
    while True:
        try:
            raw = input(f"  {prompt_label} [{default_str}]: ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            val = parser(raw)
        except ValueError:
            print(f"    [Invalid input] {error_msg}")
            continue
        if not validator(val):
            print(f"    [Out of range] {error_msg}")
            continue
        return val


def prompt_model_inputs(
    geom: RotaryGeometry,
    fuel: FuelBlendConfig,
    hybrid: HybridGeneratorConfig,
    spark_timing_deg: float,
) -> tuple[RotaryGeometry, FuelBlendConfig, HybridGeneratorConfig, float]:
    """Prompt the user for rotor radius, fuel mixture, and generator power demand."""
    display_and_refine_geometry(geom, interactive=False)
    e_mm = geom.eccentricity * 1e3

    while True:
        print("\n" + "=" * 72)
        print("ENTER SIMULATION INPUTS (Press Enter to accept [default])")
        print("=" * 72)

        r_mm = _prompt_with_default(
            f"1. Rotor radius R [mm] (e={e_mm:.1f} mm, b={geom.rotor_width*1e3:.1f} mm, CR={geom.compression_ratio:.2f})",
            geom.rotor_radius * 1e3,
            float,
            lambda v: (v / e_mm) > 3.5 and v <= 250.0,
            f"Enter R in ({3.5 * e_mm:.1f}, 250.0] mm so K = R/e > 3.5.",
            ".2f",
        )

        x_h2 = _prompt_with_default(
            f"2. Fuel mixture - H2 mass fraction x_H2 (0.00-0.30, balance gasoline, lambda={fuel.excess_air_ratio:.2f})",
            fuel.h2_mass_fraction,
            float,
            lambda v: 0.0 <= v <= 0.30,
            "Enter x_H2 in [0.00, 0.30] (e.g. 0.00 = pure gasoline, 0.10 = 10% H2).",
            ".2f",
        )

        p_req_kw = _prompt_with_default(
            "3. Power demand - Target generator electrical output P_req [kW]",
            hybrid.target_electric_power_kw,
            float,
            lambda v: 1.0 <= v <= 50.0,
            "Enter P_req in [1.0, 50.0] kW.",
            ".1f",
        )

        new_geom = RotaryGeometry(
            rotor_radius=r_mm * 1e-3,
            eccentricity=geom.eccentricity,
            rotor_width=geom.rotor_width,
            compression_ratio=geom.compression_ratio,
            recess_type=geom.recess_type,
            wall_temperature=geom.wall_temperature,
            apex_seal_clearance=geom.apex_seal_clearance,
            side_seal_clearance=geom.side_seal_clearance,
            seal_land_length=geom.seal_land_length,
        )
        new_fuel = FuelBlendConfig(h2_mass_fraction=x_h2, excess_air_ratio=fuel.excess_air_ratio)
        new_hybrid = HybridGeneratorConfig(
            target_electric_power_kw=p_req_kw,
            peak_generator_efficiency=hybrid.peak_generator_efficiency,
            rpm_min=hybrid.rpm_min,
            rpm_max=hybrid.rpm_max,
            rpm_step=hybrid.rpm_step,
            power_tolerance_kw=hybrid.power_tolerance_kw,
        )

        if abs(new_geom.rotor_radius - geom.rotor_radius) > 1e-6:
            display_and_refine_geometry(new_geom, interactive=False)
        if new_geom.compression_ratio > new_geom.theoretical_max_cr:
            print(
                f"  [Warning] CR={new_geom.compression_ratio:.2f} exceeds epitrochoid geometric "
                f"limit ({new_geom.theoretical_max_cr:.2f}) for K={new_geom.shape_factor:.2f}."
            )
        print(
            f"\n[CONFIRM INPUTS] R={r_mm:.2f} mm (Vd={new_geom.swept_volume*1e6:.2f} cm^3) | "
            f"Fuel={100*(1-x_h2):.0f}% Gas / {100*x_h2:.0f}% H2 (x_H2={x_h2:.2f}) | "
            f"P_req={p_req_kw:.1f} kW"
        )

        try:
            confirm = input("Start simulation with these inputs? [Y/n]: ").strip().lower()
        except EOFError:
            confirm = "y"
        if confirm in {"", "y", "yes"}:
            return new_geom, new_fuel, new_hybrid, spark_timing_deg
        geom, fuel, hybrid = new_geom, new_fuel, new_hybrid


def run_sanity_checks(sim_cfg: SimulationConfig, spark_timing_deg: float = 522.0) -> None:
    """Validate kinematics, thermodynamics, and ODE energy conservation before sweeps."""
    geom = sim_cfg.geometry
    th_grid = np.linspace(0.0, 1080.0, 2161)
    v_grid = chamber_volume(th_grid, geom)
    dv_num = np.gradient(v_grid, th_grid, edge_order=2)
    dv_ana = volume_derivative_deg(th_grid, geom)
    max_dv_err = float(np.max(np.abs(dv_num - dv_ana)) / np.max(np.abs(dv_ana)))

    v_min_calc = float(np.min(v_grid))
    v_max_calc = float(np.max(v_grid))
    cr_calc = v_max_calc / v_min_calc

    lookup = WankelKinematicsLookup(geom, grid_step_deg=1.0)
    _, _, a_tdc, d_hyd_tdc = lookup.evaluate(540.0)

    # Solve nominal 5000 RPM cycle and check energy balance residual
    solver = WankelCycleSolver(sim_cfg)
    test_res = solver.solve(
        OperatingConditions(rpm=5000.0, p_intake=101325.0, spark_timing_deg=spark_timing_deg)
    )

    print("\n[SANITY CHECKS]")
    print(f"  1. Kinematic dV/dtheta numerical vs. analytical max rel err : {max_dv_err:.3e}")
    print(f"  2. Verified compression ratio V_max / V_min               : {cr_calc:.4f} (target {geom.compression_ratio:.4f})")
    print(f"  3. TDC Surface Area A(540 deg) / d_hyd                    : {a_tdc*1e4:.2f} cm^2 / {d_hyd_tdc*1e3:.2f} mm")
    print(f"  4. Closed-cycle First Law energy balance rel error        : {test_res.energy_balance_rel_error:.3e}")
    print(f"  5. Baseline 5000 RPM WOT peak state                       : P_max={test_res.p_max_pa/1e5:.2f} bar, T_max={test_res.t_max_k:.0f} K")

    assert max_dv_err < 1e-3, f"Volume derivative mismatch: {max_dv_err}"
    assert abs(cr_calc - geom.compression_ratio) < 1e-3, "Compression ratio mismatch"
    assert test_res.energy_balance_rel_error < 0.03, f"Energy balance residual too large: {test_res.energy_balance_rel_error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ME401 Rotary Hybrid Range-Extender Simulation")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively for all inputs (default on TTY).")
    parser.add_argument("--batch", "--defaults", action="store_true", dest="batch", help="Run non-interactively with CLI/default values.")
    parser.add_argument("--r-mm", type=float, default=89.0, help="Rotor generating radius R [mm] (default: 89.0).")
    parser.add_argument("--e-mm", type=float, default=13.0, help="Eccentricity e [mm] (default: 13.0).")
    parser.add_argument("--b-mm", type=float, default=40.1, help="Rotor axial width b [mm] (default: 40.1).")
    parser.add_argument("--cr", type=float, default=9.55, help="Compression ratio (default: 9.55).")
    parser.add_argument("--recess", type=str, default="LDR", choices=["LDR", "SMR", "TDR"], help="Rotor recess pocket type.")
    parser.add_argument("--target-kw", type=float, default=15.0, help="Target electrical generator load [kW] (default: 15.0).")
    parser.add_argument("--h2", type=float, default=0.10, help="Hydrogen mass fraction x_H2 in [0.0, 0.30] (default: 0.10).")
    parser.add_argument("--lambda-air", type=float, default=1.15, help="Excess air ratio lambda >= 0.80 (default: 1.15).")
    parser.add_argument("--spark-deg", type=float, default=522.0, help="Spark timing [deg EA] (default: 522.0).")
    parser.add_argument("--rpm-min", type=int, default=2000, help="Minimum sweep speed [RPM] (default: 2000).")
    parser.add_argument("--rpm-max", type=int, default=7000, help="Maximum sweep speed [RPM] (default: 7000).")
    parser.add_argument("--rpm-step", type=int, default=250, help="Speed sweep increment [RPM] (default: 250).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = _resolve_output_dir()

    geom = RotaryGeometry(
        rotor_radius=args.r_mm * 1e-3,
        eccentricity=args.e_mm * 1e-3,
        rotor_width=args.b_mm * 1e-3,
        compression_ratio=args.cr,
        recess_type=args.recess,  # type: ignore[arg-type]
    )
    fuel = FuelBlendConfig(h2_mass_fraction=args.h2, excess_air_ratio=args.lambda_air)
    hybrid = HybridGeneratorConfig(
        target_electric_power_kw=args.target_kw,
        rpm_min=args.rpm_min,
        rpm_max=args.rpm_max,
        rpm_step=args.rpm_step,
    )
    spark_deg = args.spark_deg

    use_interactive = not args.batch and (args.interactive or sys.stdin.isatty())
    if use_interactive:
        geom, fuel, hybrid, spark_deg = prompt_model_inputs(geom, fuel, hybrid, spark_deg)
    else:
        geom = display_and_refine_geometry(geom, interactive=False)

    t_start = time.perf_counter()
    sim_cfg = SimulationConfig(geometry=geom, fuel=fuel, hybrid=hybrid)

    print(
        f"\n[FUEL & HYBRID CONFIG] x_H2={fuel.h2_mass_fraction:.2f} | lambda={fuel.excess_air_ratio:.2f} | "
        f"Spark={spark_deg:.1f} deg EA | LHV_mix={fuel.composite_lhv/1e6:.2f} MJ/kg | "
        f"AFR_actual={fuel.afr_actual:.2f} | P_req={hybrid.target_electric_power_kw:.1f} kW"
    )

    run_sanity_checks(sim_cfg, spark_timing_deg=spark_deg)

    print(
        f"\n[STEP 1] Running Wankel Rotary Speed Sweep "
        f"({hybrid.rpm_min:,} - {hybrid.rpm_max:,} RPM, step {hybrid.rpm_step} RPM)..."
    )
    opt_summary = run_speed_sweep_optimization(sim_cfg, spark_timing_deg=spark_deg, verbose=True)
    opt_pt = opt_summary.optimal_point

    print("\n" + "-" * 72)
    print(
        f"OPTIMAL ROTARY OPERATING POINT: N_opt = {opt_pt.rpm:.0f} RPM | "
        f"P_elec = {opt_pt.p_elec_kw:.2f} kW | P_brake = {opt_pt.p_brake_kw:.2f} kW | "
        f"BTE = {opt_pt.eta_bth*100:.2f}% | System Eff = {opt_pt.eta_system*100:.2f}% | "
        f"BSFC = {opt_pt.bsfc_g_kwh:.1f} g/(kW*h)"
    )
    print("-" * 72)

    print("\n[STEP 2] Running Equivalent Atkinson Reciprocating Piston Benchmark...")
    piston_df = run_piston_benchmark_sweep(hybrid)
    pis_feas = piston_df[piston_df["piston_feasible"]]
    if not pis_feas.empty:
        pis_opt = pis_feas.sort_values("piston_eta_system", ascending=False).iloc[0]
    else:
        pis_opt = piston_df.sort_values("piston_p_elec_wot_kw", ascending=False).iloc[0]
    print(
        f"OPTIMAL PISTON BENCHMARK POINT: N_opt = {pis_opt['rpm']:.0f} RPM | "
        f"P_elec = {pis_opt['piston_p_elec_kw']:.2f} kW | "
        f"BTE = {pis_opt['piston_eta_bth']*100:.2f}% | "
        f"System Eff = {pis_opt['piston_eta_system']*100:.2f}% | "
        f"BSFC = {pis_opt['piston_bsfc_g_kwh']:.1f} g/(kW*h)"
    )

    print("\n[STEP 3] Running Parametric Sensitivity Sweeps (CR/Recess & H2 Fraction)...")
    cr_df, h2_df = run_parametric_sensitivity(sim_cfg, spark_timing_deg=spark_deg)

    print("\n[STEP 4] Exporting CSV/JSON Results and Comparative Plots...")
    paths = export_all_outputs(sim_cfg, opt_summary, piston_df, cr_df, h2_df, base_dir)
    for k, p in paths.items():
        print(f"  Saved {k}: {p}")

    elapsed = time.perf_counter() - t_start
    print(f"\nSimulation pipeline complete in {elapsed:.2f} s.")

    if getattr(sys, "frozen", False) and not args.batch:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
