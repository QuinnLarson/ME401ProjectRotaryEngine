"""Equivalent standard gasoline reciprocating piston engine benchmark model.

Models an equivalent 4-stroke reciprocating Atkinson-cycle gasoline range-extender
baseline meeting the identical 15 kW electrical load across 2,000 - 7,000 RPM.

Sizing & Equivalence Rationale:
- Per Boughou & Mohiuddin (2020, §3), a single-rotor Wankel engine has three
  chambers executing a 1080-deg cycle, producing 1 combustion power stroke per
  shaft revolution. A 2-cylinder 4-stroke reciprocating engine (720-deg cycle)
  likewise produces 1 power stroke per crankshaft revolution.
- We size the 2-cylinder baseline with bore = 68.0 mm, stroke = 66.3 mm
  (240.8 cm^3 per cylinder, 481.6 cm^3 total displacement — matching the BMW i3
  / Toyota-style compact twin-cylinder range-extender architecture cited in
  `PROJECT_GUIDELINES.md` §5 and `CombustionChamberDesignEffect-RotaryEngine.pdf`).
- Uses geometric expansion ratio CR_exp = 12.2 with delayed intake valve closing
  (LIVC at 228 deg CA) to realize an Atkinson over-expansion cycle.
- Integrates the full 720-deg open-system thermodynamic ODE using the exact same
  `MixtureThermoModel` NASA polynomials, standard Woschni piston heat transfer,
  piston-ring blow-by, Chen-Flynn rubbing FMEP, and the identical PMSM generator
  efficiency curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import numpy as np
from pandas import DataFrame
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from combustion import compute_blended_flame_speed
from config import FuelBlendConfig, HybridGeneratorConfig
from submodels import orifice_mass_flow_rate_deg, port_open_fraction
from thermo_properties import MixtureThermoModel


@dataclass
class PistonEngineGeometry:
    """2-cylinder 4-stroke Atkinson-cycle reciprocating piston engine geometry."""

    n_cylinders: int = 2
    bore_m: float = 0.0680
    stroke_m: float = 0.0663
    conrod_m: float = 0.1160  # R_rod / a_crank = 3.5
    cr_expansion: float = 12.2
    ivc_deg: float = 228.0  # Late Intake Valve Closing (Atkinson over-expansion)
    evo_deg: float = 505.0  # Exhaust Valve Opening
    wall_temperature_k: float = 415.0
    ring_clearance_m: float = 6.0e-6

    @property
    def crank_radius_m(self) -> float:
        return 0.5 * self.stroke_m

    @property
    def swept_volume_cyl(self) -> float:
        return 0.25 * math.pi * (self.bore_m ** 2) * self.stroke_m

    @property
    def min_volume_cyl(self) -> float:
        return self.swept_volume_cyl / (self.cr_expansion - 1.0)


def piston_kinematics(theta_ca_deg: float, geom: PistonEngineGeometry) -> tuple[float, float, float, float]:
    """Slider-crank V(theta), dV/dtheta_deg, exposed area A(theta), and d_hyd [m].

    Angle convention (720-deg 4-stroke cycle):
        0 - 180 deg:   Intake (TDC -> BDC)
        180 - 360 deg: Compression (BDC -> TDC firing at 360 deg)
        360 - 540 deg: Expansion (TDC -> BDC)
        540 - 720 deg: Exhaust (BDC -> TDC)
    """
    th = math.radians(theta_ca_deg)
    a = geom.crank_radius_m
    l_rod = geom.conrod_m
    lam = a / l_rod

    sin_t = math.sin(th)
    cos_t = math.cos(th)
    root = math.sqrt(max(1.0 - (lam * sin_t) ** 2, 1e-8))

    # Instantaneous displacement from TDC
    s = a * (1.0 - cos_t) + l_rod * (1.0 - root)
    ds_dth_rad = a * sin_t * (1.0 + (lam * cos_t) / root)

    area_piston = 0.25 * math.pi * (geom.bore_m ** 2)
    v = geom.min_volume_cyl + area_piston * s
    dv_ddeg = area_piston * ds_dth_rad * (math.pi / 180.0)

    # Exposed surface area: piston crown (~1.05 * A_p) + cylinder head (~1.12 * A_p) + liner
    clearance_h = geom.min_volume_cyl / area_piston
    liner_h = clearance_h + s
    a_total = (1.05 + 1.12) * area_piston + math.pi * geom.bore_m * liner_h
    d_hyd = 4.0 * v / max(a_total, 1e-9)
    return v, dv_ddeg, a_total, d_hyd


class ReciprocatingBenchmarkSolver:
    """720-deg stiff ODE solver for the baseline gasoline Atkinson piston engine."""

    def __init__(self, geom: PistonEngineGeometry | None = None) -> None:
        self.geom = geom or PistonEngineGeometry()
        self.fuel = FuelBlendConfig(h2_mass_fraction=0.0, excess_air_ratio=1.0)
        self.thermo = MixtureThermoModel(self.fuel)
        # Valve effective flow areas per cylinder
        area_p = 0.25 * math.pi * (self.geom.bore_m ** 2)
        self.a_in_max = 0.14 * area_p
        self.a_ex_max = 0.12 * area_p

    def _solve_single_pass(
        self,
        rpm: float,
        p_intake_pa: float,
        p0_pa: float,
        t0_k: float,
        m_fuel_est: float,
        p_ivc_ref: float,
        t_ivc_ref: float,
        v_ivc_ref: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        t_in = 305.0
        p_exh = 105000.0
        spark_deg = 344.0  # 16 deg BTDC relative to 360 deg TDC

        # Wiebe burn duration in 720-deg crank angle (~44 deg CA at 5000 RPM)
        s_l_ref = compute_blended_flame_speed(680.0, 1.8e6, self.fuel)
        mean_piston_speed = 2.0 * self.geom.stroke_m * (rpm / 60.0)
        mean_piston_ref = 2.0 * self.geom.stroke_m * (5000.0 / 60.0)
        s_t_ref = s_l_ref + 0.55 * mean_piston_ref
        s_t = s_l_ref + 0.55 * mean_piston_speed
        dur_ca = max(28.0, min(68.0, 44.0 * (rpm / 5000.0) * (s_t_ref / s_t)))
        eta_comb = 0.975  # Compact cylindrical chamber minimizes wall quenching

        v0, _, _, _ = piston_kinematics(0.0, self.geom)
        gas0 = self.thermo.evaluate(t0_k, burned_fraction=1.0, residual_fraction=1.0)
        m0 = p0_pa * v0 / (gas0.r_specific * t0_k)
        y0 = np.array([p0_pa, t0_k, m0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        theta_eval = np.arange(0.0, 720.0 + 0.5, 0.5)

        gas_in = self.thermo.evaluate(t_in, burned_fraction=0.0, residual_fraction=0.0)
        gas_ex = self.thermo.evaluate(780.0, burned_fraction=1.0, residual_fraction=1.0)
        v_ivc_geom, _, _, _ = piston_kinematics(self.geom.ivc_deg, self.geom)

        def make_rhs(
            c2_val: float,
            c3_val: float,
            m_fuel_val: float,
            p_ivc_val: float,
            t_ivc_val: float,
            v_ivc_val: float,
        ):
            woschni_comb_coeff = c3_val * (
                self.geom.swept_volume_cyl * t_ivc_val / max(p_ivc_val * v_ivc_val, 1e-6)
            )
            ring_geom_coeff = math.pi * self.geom.bore_m * (self.geom.ring_clearance_m ** 3) / (12.0 * 0.003)

            def rhs(theta: float, y: np.ndarray) -> np.ndarray:
                p = max(float(y[0]), 5000.0)
                t = max(float(y[1]), 200.0)
                m = max(float(y[2]), 1e-8)
                m_fresh = max(float(y[3]), 0.0)

                v, dv_ddeg, area, d_hyd = piston_kinematics(theta, self.geom)

                # Wiebe progress
                if theta <= spark_deg or m_fuel_val <= 0.0:
                    xb = 0.0
                    dxb = 0.0
                else:
                    tau = (theta - spark_deg) / dur_ca
                    if tau >= 1.4:
                        xb = 1.0
                        dxb = 0.0
                    else:
                        a_w, m_w = 5.0, 2.0
                        exp_term = math.exp(-a_w * (tau ** (m_w + 1.0)))
                        xb = 1.0 - exp_term
                        dxb = a_w * (m_w + 1.0) / dur_ca * (tau ** m_w) * exp_term

                res_frac = max(0.02, min(1.0, 1.0 - m_fresh / m))
                gas = self.thermo.evaluate(t, burned_fraction=xb, residual_fraction=res_frac)

                # Valve flows (intake 0 -> ivc_deg, exhaust evo_deg -> 720)
                in_frac = port_open_fraction(theta, 0.0, self.geom.ivc_deg, ramp_deg=15.0)
                ex_frac = port_open_fraction(theta, self.geom.evo_deg, 720.0, ramp_deg=15.0)
                a_in = self.a_in_max * in_frac
                a_ex = self.a_ex_max * ex_frac

                dm_in_f = orifice_mass_flow_rate_deg(p_intake_pa, t_in, p, a_in, gas_in.gamma, gas_in.r_specific, rpm)
                dm_in_b = orifice_mass_flow_rate_deg(p, t, p_intake_pa, a_in, gas.gamma, gas.r_specific, rpm)
                dm_ex_f = orifice_mass_flow_rate_deg(p, t, p_exh, a_ex, gas.gamma, gas.r_specific, rpm)
                dm_ex_b = orifice_mass_flow_rate_deg(p_exh, 780.0, p, a_ex, gas_ex.gamma, gas_ex.r_specific, rpm)

                # Heat release
                dq_comb = m_fuel_val * eta_comb * self.fuel.composite_lhv * dxb

                # Standard Woschni heat transfer for reciprocating engine
                p_mot = p_ivc_val * ((v_ivc_val / v) ** 1.34)
                w_vel = max(0.5, c2_val * mean_piston_speed + woschni_comb_coeff * max(0.0, p - p_mot))
                h_c = 3.26 * (max(d_hyd, 0.004) ** -0.2) * ((p / 1000.0) ** 0.8) * (t ** -0.53) * (w_vel ** 0.8)
                dq_wall = h_c * area * (t - self.geom.wall_temperature_k) / (6.0 * rpm)

                # Piston ring blow-by (smooth above atmospheric pressure)
                dp_seal = p - 101325.0
                if dp_seal <= 0.0 or m <= 1.0e-7:
                    dm_leak = 0.0
                else:
                    rho = p / max(gas.r_specific * t, 1e-3)
                    q_ring = (ring_geom_coeff * dp_seal) / gas.viscosity
                    dm_leak = (rho * q_ring) / (6.0 * rpm)

                dm_ddeg = (dm_in_f - dm_in_b) - (dm_ex_f - dm_ex_b) - dm_leak
                dm_fresh_ddeg = dm_in_f - (m_fresh / m) * (dm_in_b + dm_ex_f + dm_leak)

                u_ch = gas.enthalpy - gas.r_specific * t
                h_in_term = (gas_in.enthalpy - u_ch) * dm_in_f - (gas.r_specific * t) * dm_in_b
                h_ex_term = -(gas.r_specific * t) * dm_ex_f + (gas_ex.enthalpy - u_ch) * dm_ex_b
                h_leak_term = -(gas.r_specific * t) * dm_leak

                dt_ddeg = (dq_comb - dq_wall - p * dv_ddeg + h_in_term + h_ex_term + h_leak_term) / max(m * gas.cv, 1e-9)
                dp_ddeg = p * ((dm_ddeg / m) + (dt_ddeg / t) - (dv_ddeg / v))

                return np.array([dp_ddeg, dt_ddeg, dm_ddeg, dm_fresh_ddeg, dq_comb, dq_wall, dm_leak], dtype=float)

            return rhs

        def integrate_segment(
            t_start: float,
            t_end: float,
            y_init: np.ndarray,
            t_eval_seg: np.ndarray,
            rhs_fn,
        ) -> tuple[np.ndarray, np.ndarray]:
            need_dense = len(t_eval_seg) == 0 or abs(float(t_eval_seg[-1]) - t_end) > 1e-9
            t_eval_arg = t_eval_seg if len(t_eval_seg) > 0 else None
            sol = solve_ivp(
                rhs_fn,
                (t_start, t_end),
                y_init,
                method="BDF",
                t_eval=t_eval_arg,
                dense_output=need_dense,
                max_step=0.5,
                rtol=1e-6,
                atol=1e-8,
            )
            if not sol.success:
                sol = solve_ivp(
                    rhs_fn,
                    (t_start, t_end),
                    y_init,
                    method="Radau",
                    t_eval=t_eval_arg,
                    dense_output=need_dense,
                    max_step=0.5,
                    rtol=1e-6,
                    atol=1e-8,
                )
            if not sol.success:
                raise RuntimeError(f"Piston benchmark ODE failed at {rpm} RPM [{t_start}, {t_end}]: {sol.message}")
            y_end = sol.sol(t_end) if need_dense else sol.y[:, -1].copy()
            y_out = sol.y if len(t_eval_seg) > 0 else np.empty((len(y_init), 0), dtype=float)
            return y_out, y_end

        ivc_deg = self.geom.ivc_deg
        evo_deg = self.geom.evo_deg

        # Segment 1: Intake (0 -> IVC), gas exchange C2 = 6.18, zero combustion
        mask1 = theta_eval <= ivc_deg
        y1_seg, y_ivc = integrate_segment(
            0.0,
            ivc_deg,
            y0,
            theta_eval[mask1],
            make_rhs(6.18, 0.0, 0.0, p_ivc_ref, t_ivc_ref, v_ivc_ref),
        )
        p_ivc = max(float(y_ivc[0]), 5000.0)
        t_ivc = max(float(y_ivc[1]), 200.0)
        m_fresh_ivc = max(float(y_ivc[3]), 0.0)
        m_fuel_locked = m_fresh_ivc / (1.0 + self.fuel.afr_actual) if m_fresh_ivc > 1e-9 else m_fuel_est

        # Segment 2: Closed compression before spark (IVC -> spark_deg)
        mask2 = (theta_eval > ivc_deg) & (theta_eval <= spark_deg)
        y2_seg, y_spk = integrate_segment(
            ivc_deg,
            spark_deg,
            y_ivc,
            theta_eval[mask2],
            make_rhs(2.28, 0.0, 0.0, p_ivc, t_ivc, v_ivc_geom),
        )

        # Segment 3: Combustion & closed expansion (spark_deg -> EVO)
        mask3 = (theta_eval > spark_deg) & (theta_eval <= evo_deg)
        y3_seg, y_evo = integrate_segment(
            spark_deg,
            evo_deg,
            y_spk,
            theta_eval[mask3],
            make_rhs(2.28, 0.00324, m_fuel_locked, p_ivc, t_ivc, v_ivc_geom),
        )

        # Segment 4: Exhaust blowdown & pumping (EVO -> 720 deg)
        mask4 = theta_eval > evo_deg
        y4_seg, _ = integrate_segment(
            evo_deg,
            720.0,
            y_evo,
            theta_eval[mask4],
            make_rhs(6.18, 0.0, m_fuel_locked, p_ivc, t_ivc, v_ivc_geom),
        )

        y_full = np.hstack([y1_seg, y2_seg, y3_seg, y4_seg])
        return theta_eval, y_full, m_fuel_locked, p_ivc, t_ivc

    def solve_cycle(self, rpm: float, p_intake_pa: float = 101325.0) -> Dict[str, float]:
        """Solve converged 2-cylinder 720-deg Atkinson cycle and return performance metrics."""
        gas_in = self.thermo.evaluate(305.0, 0.0, 0.0)
        rho_in = p_intake_pa / (gas_in.r_specific * 305.0)
        v_ivc_est, _, _, _ = piston_kinematics(self.geom.ivc_deg, self.geom)
        m_fuel_est = 0.78 * rho_in * self.geom.swept_volume_cyl / (1.0 + self.fuel.afr_actual)

        t1, y1, mf1, p_ivc1, t_ivc1 = self._solve_single_pass(
            rpm, p_intake_pa, 105000.0, 820.0, m_fuel_est, p_intake_pa, 305.0, v_ivc_est
        )
        theta, y, _, _, _ = self._solve_single_pass(
            rpm, p_intake_pa, float(y1[0, -1]), float(np.clip(y1[1, -1], 450.0, 1300.0)),
            mf1, p_ivc1, t_ivc1, v_ivc_est
        )

        p_arr = y[0]
        v_arr = np.array([piston_kinematics(float(th), self.geom)[0] for th in theta])
        idx_ivc = int(np.argmin(np.abs(theta - self.geom.ivc_deg)))
        m_fresh_ivc = float(y[3, idx_ivc])
        m_fuel_cyl = max(1e-10, m_fresh_ivc / (1.0 + self.fuel.afr_actual))

        trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        w_ind_cyl = float(trapz_fn(p_arr, v_arr))
        mask_hp = (theta >= 180.0) & (theta <= 540.0)
        w_gross_cyl = float(trapz_fn(p_arr[mask_hp], v_arr[mask_hp]))
        w_pump_cyl = max(0.0, -(w_ind_cyl - w_gross_cyl))

        # Chen-Flynn reciprocating FMEP [Pa] (Heywood 2018)
        p_max_bar = float(np.max(p_arr)) / 1e5
        s_p = 2.0 * self.geom.stroke_m * (rpm / 60.0)
        fmep_bar = 0.40 + 0.005 * p_max_bar + 0.042 * s_p + 0.0009 * (s_p ** 2)
        w_fric_cyl = fmep_bar * 1e5 * self.geom.swept_volume_cyl

        w_brake_cyl = max(0.0, w_ind_cyl - w_fric_cyl)
        # 2 cylinders firing once every 720 deg -> 1 cycle/rev = (rpm / 60) cycles/s
        cycles_per_s = rpm / 60.0
        p_ind_kw = (w_ind_cyl * cycles_per_s) / 1000.0
        p_brake_kw = (w_brake_cyl * cycles_per_s) / 1000.0

        q_fuel_cyl = m_fuel_cyl * self.fuel.composite_lhv
        eta_ith = w_ind_cyl / max(q_fuel_cyl, 1e-9)
        eta_bth = w_brake_cyl / max(q_fuel_cyl, 1e-9)
        fuel_g_hr = m_fuel_cyl * cycles_per_s * 3600.0 * 1000.0
        bsfc = fuel_g_hr / p_brake_kw if p_brake_kw > 0.05 else float("inf")

        m_ivc = float(y[2, idx_ivc])
        m_leak = max(0.0, float(y[6, -1] - y[6, idx_ivc]))
        return {
            "rpm": rpm,
            "p_intake_kpa": p_intake_pa / 1000.0,
            "p_ind_kw": p_ind_kw,
            "p_brake_kw": p_brake_kw,
            "eta_ith": eta_ith,
            "eta_bth": eta_bth,
            "bsfc_g_kwh": bsfc,
            "q_wall_pct": 100.0 * float(y[5, -1]) / max(q_fuel_cyl, 1e-9),
            "leak_mass_pct": 100.0 * m_leak / max(m_ivc, 1e-9),
            "friction_pct": 100.0 * w_fric_cyl / max(q_fuel_cyl, 1e-9),
            "pumping_pct": 100.0 * w_pump_cyl / max(q_fuel_cyl, 1e-9),
            "p_max_bar": p_max_bar,
            "t_max_k": float(np.max(y[1])),
        }


def run_piston_benchmark_sweep(hybrid_cfg: HybridGeneratorConfig) -> DataFrame:
    """Evaluate equivalent Atkinson reciprocating piston baseline across the RPM sweep."""
    solver = ReciprocatingBenchmarkSolver()
    rpms = list(range(hybrid_cfg.rpm_min, hybrid_cfg.rpm_max + 1, hybrid_cfg.rpm_step))
    p_req = hybrid_cfg.target_electric_power_kw
    tol = hybrid_cfg.power_tolerance_kw

    rows: List[Dict[str, float | bool]] = []
    for rpm in rpms:
        rpm_f = float(rpm)
        wot = solver.solve_cycle(rpm_f, 101325.0)
        eta_g_wot = hybrid_cfg.generator_efficiency(rpm_f, wot["p_brake_kw"])
        p_elec_wot = wot["p_brake_kw"] * eta_g_wot

        if p_elec_wot < p_req - tol:
            chosen = wot
            feasible = False
        elif abs(p_elec_wot - p_req) <= tol:
            chosen = wot
            feasible = True
        else:
            cache: Dict[float, Dict[str, float]] = {101325.0: wot}

            def res_fn(p_in: float) -> float:
                out = cache.get(p_in)
                if out is None:
                    out = solver.solve_cycle(rpm_f, p_in)
                    cache[p_in] = out
                return out["p_brake_kw"] * hybrid_cfg.generator_efficiency(rpm_f, out["p_brake_kw"]) - p_req

            if res_fn(35000.0) > 0.0:
                chosen = cache[35000.0]
            else:
                p_match = brentq(res_fn, 35000.0, 101325.0, xtol=350.0, rtol=5e-3, maxiter=12)
                chosen = cache.get(p_match) or solver.solve_cycle(rpm_f, float(p_match))
            feasible = True

        eta_g = hybrid_cfg.generator_efficiency(rpm_f, chosen["p_brake_kw"])
        p_elec = chosen["p_brake_kw"] * eta_g
        status = "FEASIBLE" if feasible else "UNDER-PWR"
        print(
            f"  [Piston] RPM={rpm:4d} | {status:9s} | P_wot={p_elec_wot:5.2f} kW | "
            f"P_elec={p_elec:5.2f} kW | BTE={chosen['eta_bth']*100:5.2f}% | "
            f"BSFC={chosen['bsfc_g_kwh']:6.1f} g/kWh"
        )
        rows.append(
            {
                "rpm": rpm_f,
                "piston_feasible": feasible,
                "piston_p_elec_wot_kw": p_elec_wot,
                "piston_p_brake_kw": chosen["p_brake_kw"],
                "piston_p_elec_kw": p_elec,
                "piston_eta_gen": eta_g,
                "piston_eta_ith": chosen["eta_ith"],
                "piston_eta_bth": chosen["eta_bth"],
                "piston_eta_system": chosen["eta_bth"] * eta_g,
                "piston_bsfc_g_kwh": chosen["bsfc_g_kwh"],
                "piston_q_wall_pct": chosen["q_wall_pct"],
                "piston_leak_mass_pct": chosen["leak_mass_pct"],
                "piston_friction_pct": chosen["friction_pct"],
                "piston_pumping_pct": chosen["pumping_pct"],
                "piston_p_max_bar": chosen["p_max_bar"],
                "piston_t_max_k": chosen["t_max_k"],
            }
        )

    return DataFrame(rows)
