"""Speed sweep optimizer and parametric sensitivity framework for rotary APU.

Implements:
1. Speed sweep across 2,000 - 7,000 RPM in 250-RPM increments.
2. Evaluates both WOT (wide-open throttle) capability and load-matched throttled
   operation meeting the target generator electrical load (P_req = 15 kW default).
3. Selects optimal operating RPM (N_opt) maximizing brake/system thermal efficiency
   (and minimizing BSFC) subject to P_elec == P_req +/- epsilon.
4. Executes parametric sweeps over compression ratio (CR = 8.33, 9.55, 10.18),
   recess geometry (LDR vs SMR), and hydrogen mass fraction (x_H2 = 0.0 - 0.30).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import FuelBlendConfig, OperatingConditions, RecessType, RotaryGeometry, SimulationConfig
from cycle_solver import CycleResults, WankelCycleSolver


@dataclass
class SweepPointRecord:
    """Structured performance record at one RPM in the speed sweep."""

    rpm: float
    feasible: bool
    p_intake_wot_kpa: float
    p_brake_wot_kw: float
    p_elec_wot_kw: float
    p_intake_op_kpa: float
    p_ind_kw: float
    p_brake_kw: float
    p_elec_kw: float
    eta_gen: float
    eta_ith: float
    eta_bth: float
    eta_system: float
    bsfc_g_kwh: float
    imep_net_bar: float
    bmep_bar: float
    fmep_total_bar: float
    fmep_rubbing_bar: float
    fmep_seal_drag_bar: float
    pmep_bar: float
    volumetric_eff: float
    p_max_bar: float
    t_max_k: float
    q_wall_loss_pct: float
    leak_mass_pct: float
    h_leak_loss_pct: float
    friction_loss_pct: float
    pumping_loss_pct: float
    exhaust_and_other_pct: float
    energy_balance_rel_error: float


@dataclass
class OptimizationSummary:
    """Full RPM sweep results and optimal operating point."""

    sweep_df: pd.DataFrame
    optimal_point: SweepPointRecord
    optimal_cycle: CycleResults
    wot_cycles: Dict[float, CycleResults]
    matched_cycles: Dict[float, CycleResults]


def _extract_record(
    res: CycleResults,
    wot_res: CycleResults,
    sim_cfg: SimulationConfig,
    feasible: bool,
) -> SweepPointRecord:
    """Convert a CycleResults object into a flat SweepPointRecord with loss percentages."""
    eta_gen = sim_cfg.hybrid.generator_efficiency(res.rpm, res.p_brake_kw)
    p_elec = res.p_brake_kw * eta_gen
    eta_gen_wot = sim_cfg.hybrid.generator_efficiency(wot_res.rpm, wot_res.p_brake_kw)
    p_elec_wot = wot_res.p_brake_kw * eta_gen_wot

    q_in = max(res.q_fuel_input_j, 1e-9)
    q_wall_pct = 100.0 * res.q_wall_loss_j / q_in
    h_leak_pct = 100.0 * (res.h_leak_loss_j / q_in + res.leak_mass_pct / 100.0 * 0.65)
    fric_pct = 100.0 * res.mechanical_losses.work_friction_total_j / q_in
    pump_pct = 100.0 * res.work_pumping_j / q_in
    bte_pct = 100.0 * res.eta_bth
    exhaust_other_pct = max(0.0, 100.0 - bte_pct - q_wall_pct - h_leak_pct - fric_pct - pump_pct)

    return SweepPointRecord(
        rpm=res.rpm,
        feasible=feasible,
        p_intake_wot_kpa=wot_res.p_intake_pa / 1000.0,
        p_brake_wot_kw=wot_res.p_brake_kw,
        p_elec_wot_kw=p_elec_wot,
        p_intake_op_kpa=res.p_intake_pa / 1000.0,
        p_ind_kw=res.p_ind_kw,
        p_brake_kw=res.p_brake_kw,
        p_elec_kw=p_elec,
        eta_gen=eta_gen,
        eta_ith=res.eta_ith,
        eta_bth=res.eta_bth,
        eta_system=res.eta_bth * eta_gen,
        bsfc_g_kwh=res.bsfc_g_kwh,
        imep_net_bar=res.imep_net_pa / 1e5,
        bmep_bar=res.bmep_pa / 1e5,
        fmep_total_bar=res.mechanical_losses.fmep_total_pa / 1e5,
        fmep_rubbing_bar=res.mechanical_losses.fmep_rubbing_pa / 1e5,
        fmep_seal_drag_bar=res.mechanical_losses.fmep_seal_drag_pa / 1e5,
        pmep_bar=res.mechanical_losses.pmep_pa / 1e5,
        volumetric_eff=res.volumetric_efficiency,
        p_max_bar=res.p_max_pa / 1e5,
        t_max_k=res.t_max_k,
        q_wall_loss_pct=q_wall_pct,
        leak_mass_pct=res.leak_mass_pct,
        h_leak_loss_pct=h_leak_pct,
        friction_loss_pct=fric_pct,
        pumping_loss_pct=pump_pct,
        exhaust_and_other_pct=exhaust_other_pct,
        energy_balance_rel_error=res.energy_balance_rel_error,
    )


def solve_load_matched_cycle(
    solver: WankelCycleSolver,
    rpm: float,
    sim_cfg: SimulationConfig,
    wot_res: CycleResults,
    spark_timing_deg: float = 522.0,
) -> tuple[CycleResults, bool]:
    """Match generator electrical demand P_req via throttle manifold pressure P_intake.

    Returns (CycleResults, is_feasible). If WOT electrical power < P_req - tol,
    returns (wot_res, False).
    """
    p_req = sim_cfg.hybrid.target_electric_power_kw
    tol = sim_cfg.hybrid.power_tolerance_kw

    eta_gen_wot = sim_cfg.hybrid.generator_efficiency(rpm, wot_res.p_brake_kw)
    p_elec_wot = wot_res.p_brake_kw * eta_gen_wot

    if p_elec_wot < p_req - tol:
        return wot_res, False
    if abs(p_elec_wot - p_req) <= tol:
        return wot_res, True

    # Cache evaluations to avoid duplicate ODE solves at endpoint
    eval_cache: Dict[float, CycleResults] = {101325.0: wot_res}

    def power_residual(p_in_pa: float) -> float:
        res = eval_cache.get(p_in_pa)
        if res is None:
            op = OperatingConditions(
                rpm=rpm,
                p_intake=p_in_pa,
                spark_timing_deg=spark_timing_deg,
            )
            res = solver.solve(op)
            eval_cache[p_in_pa] = res
        eta_g = sim_cfg.hybrid.generator_efficiency(rpm, res.p_brake_kw)
        return res.p_brake_kw * eta_g - p_req

    p_low = 35000.0
    if power_residual(p_low) > 0.0:
        return eval_cache[p_low], True

    p_opt_in = brentq(power_residual, p_low, 101325.0, xtol=350.0, rtol=5e-3, maxiter=12)
    matched_res = eval_cache.get(p_opt_in)
    if matched_res is None:
        matched_res = solver.solve(
            OperatingConditions(rpm=rpm, p_intake=float(p_opt_in), spark_timing_deg=spark_timing_deg)
        )
    return matched_res, True


def run_speed_sweep_optimization(
    sim_cfg: SimulationConfig,
    spark_timing_deg: float = 522.0,
    verbose: bool = True,
) -> OptimizationSummary:
    """Sweep 2,000 - 7,000 RPM in 250-RPM increments and find optimal N_opt."""
    solver = WankelCycleSolver(sim_cfg)
    rpms = list(range(sim_cfg.hybrid.rpm_min, sim_cfg.hybrid.rpm_max + 1, sim_cfg.hybrid.rpm_step))

    records: List[SweepPointRecord] = []
    wot_cycles: Dict[float, CycleResults] = {}
    matched_cycles: Dict[float, CycleResults] = {}

    for rpm in rpms:
        rpm_f = float(rpm)
        wot_op = OperatingConditions(rpm=rpm_f, p_intake=101325.0, spark_timing_deg=spark_timing_deg)
        wot_res = solver.solve(wot_op)
        wot_cycles[rpm_f] = wot_res

        matched_res, feasible = solve_load_matched_cycle(
            solver, rpm_f, sim_cfg, wot_res, spark_timing_deg=spark_timing_deg
        )
        matched_cycles[rpm_f] = matched_res
        rec = _extract_record(matched_res, wot_res, sim_cfg, feasible)
        records.append(rec)

        if verbose:
            status = "FEASIBLE" if feasible else "UNDER-PWR"
            print(
                f"  RPM={rpm:4d} | {status:9s} | P_wot={rec.p_elec_wot_kw:5.2f} kW | "
                f"P_elec={rec.p_elec_kw:5.2f} kW | BTE={rec.eta_bth*100:5.2f}% | "
                f"BSFC={rec.bsfc_g_kwh:6.1f} g/kWh | Leak={rec.leak_mass_pct:4.2f}% | "
                f"Qwall={rec.q_wall_loss_pct:4.1f}%"
            )

    df = pd.DataFrame([r.__dict__ for r in records])
    feasible_records = [r for r in records if r.feasible]

    if feasible_records:
        # Maximize overall system efficiency (eta_bth * eta_gen), equivalently minimize BSFC
        optimal_rec = max(feasible_records, key=lambda r: r.eta_system)
    else:
        optimal_rec = max(records, key=lambda r: r.p_elec_wot_kw)

    optimal_cycle = matched_cycles[optimal_rec.rpm]
    return OptimizationSummary(
        sweep_df=df,
        optimal_point=optimal_rec,
        optimal_cycle=optimal_cycle,
        wot_cycles=wot_cycles,
        matched_cycles=matched_cycles,
    )


def run_parametric_sensitivity(
    base_cfg: SimulationConfig,
    rpms: Optional[List[int]] = None,
    spark_timing_deg: float = 522.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run parametric sweeps for (1) CR & Recess (LDR vs SMR) and (2) H2 mass fraction x_H2."""
    if rpms is None:
        rpms = [3000, 4000, 4750, 5000, 5500, 6000, 7000]

    # 1. Compression Ratio & Recess Pocket Sweep (Boughou & Mohiuddin 2020)
    cr_rows = []
    for recess in ("LDR", "SMR"):
        for cr in (8.33, 9.55, 10.18):
            print(f"  [Sensitivity CR] Recess={recess}, CR={cr:.2f} ...")
            geom = RotaryGeometry(
                rotor_radius=base_cfg.geometry.rotor_radius,
                eccentricity=base_cfg.geometry.eccentricity,
                rotor_width=base_cfg.geometry.rotor_width,
                compression_ratio=cr,
                recess_type=recess,  # type: ignore[arg-type]
                wall_temperature=base_cfg.geometry.wall_temperature,
                apex_seal_clearance=base_cfg.geometry.apex_seal_clearance,
                side_seal_clearance=base_cfg.geometry.side_seal_clearance,
                seal_land_length=base_cfg.geometry.seal_land_length,
            )
            cfg = SimulationConfig(geometry=geom, fuel=base_cfg.fuel, hybrid=base_cfg.hybrid)
            solver = WankelCycleSolver(cfg)
            for rpm in rpms:
                wot_res = solver.solve(
                    OperatingConditions(
                        rpm=float(rpm),
                        p_intake=101325.0,
                        spark_timing_deg=spark_timing_deg,
                    )
                )
                matched_res, feasible = solve_load_matched_cycle(
                    solver, float(rpm), cfg, wot_res, spark_timing_deg=spark_timing_deg
                )
                eta_g = cfg.hybrid.generator_efficiency(float(rpm), matched_res.p_brake_kw)
                cr_rows.append(
                    {
                        "recess_type": recess,
                        "CR": cr,
                        "rpm": rpm,
                        "feasible": feasible,
                        "p_elec_wot_kw": wot_res.p_brake_kw * cfg.hybrid.generator_efficiency(float(rpm), wot_res.p_brake_kw),
                        "p_elec_kw": matched_res.p_brake_kw * eta_g,
                        "eta_bth_pct": matched_res.eta_bth * 100.0,
                        "eta_sys_pct": matched_res.eta_bth * eta_g * 100.0,
                        "bsfc_g_kwh": matched_res.bsfc_g_kwh,
                        "p_max_bar": matched_res.p_max_pa / 1e5,
                        "leak_mass_pct": matched_res.leak_mass_pct,
                    }
                )

    # 2. Hydrogen Blending Sensitivity Sweep (x_H2 in [0.0, 0.1, 0.2, 0.3], Yang et al. 2022)
    h2_rows = []
    for x_h2 in (0.0, 0.10, 0.20, 0.30):
        print(f"  [Sensitivity H2] x_H2={x_h2:.2f} ...")
        fuel = FuelBlendConfig(h2_mass_fraction=x_h2, excess_air_ratio=base_cfg.fuel.excess_air_ratio)
        cfg = SimulationConfig(geometry=base_cfg.geometry, fuel=fuel, hybrid=base_cfg.hybrid)
        solver = WankelCycleSolver(cfg)
        for rpm in rpms:
            wot_res = solver.solve(
                OperatingConditions(
                    rpm=float(rpm),
                    p_intake=101325.0,
                    spark_timing_deg=spark_timing_deg,
                )
            )
            matched_res, feasible = solve_load_matched_cycle(
                solver, float(rpm), cfg, wot_res, spark_timing_deg=spark_timing_deg
            )
            eta_g = cfg.hybrid.generator_efficiency(float(rpm), matched_res.p_brake_kw)
            # Indicated CO2 emission [g / (kW*h)] (Yang et al. 2022 Eq. 24)
            co2_per_fuel = fuel.carbon_mass_fraction * (44.01 / 12.011)
            brake_co2_g_kwh = matched_res.bsfc_g_kwh * co2_per_fuel
            h2_rows.append(
                {
                    "x_H2": x_h2,
                    "lambda": fuel.excess_air_ratio,
                    "rpm": rpm,
                    "feasible": feasible,
                    "p_elec_wot_kw": wot_res.p_brake_kw * cfg.hybrid.generator_efficiency(float(rpm), wot_res.p_brake_kw),
                    "p_elec_kw": matched_res.p_brake_kw * eta_g,
                    "eta_bth_pct": matched_res.eta_bth * 100.0,
                    "eta_sys_pct": matched_res.eta_bth * eta_g * 100.0,
                    "bsfc_g_kwh": matched_res.bsfc_g_kwh,
                    "co2_g_kwh": brake_co2_g_kwh,
                    "burn_duration_deg": matched_res.combustion_params.duration_deg,
                    "t_max_k": matched_res.t_max_k,
                }
            )

    return pd.DataFrame(cr_rows), pd.DataFrame(h2_rows)

