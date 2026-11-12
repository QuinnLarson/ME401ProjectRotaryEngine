"""Zero-dimensional open-system thermodynamic cycle solver for Wankel rotary engines.

Integrates the coupled differential conservation equations across the complete
1080-deg eccentric shaft 4-stroke cycle using `scipy.integrate.solve_ivp` with
implicit stiff solvers ('Radau' with 'BDF' fallback), max_step <= 0.5 deg,
rtol = 1e-6, and atol = 1e-8.

State vector y(theta) over theta in [0, 1080] deg:
    y = [P, T, m, m_fresh, Q_comb_cum, Q_wall_cum, m_leak_cum, H_leak_cum]

Governing equations (First Law & Mass Conservation):
    dm/dtheta = dm_in/dtheta - dm_exh/dtheta - dm_leak/dtheta
    m*Cv*(dT/dtheta) = dQ_comb/dtheta - dQ_wall/dtheta - P*(dV/dtheta)
                       + (h_in - u)*dm_in/dtheta - R_spec*T*(dm_exh/dtheta + dm_leak/dtheta)
    dP/dtheta = P * ( (1/m)*(dm/dtheta) + (1/T)*(dT/dtheta) - (1/V)*(dV/dtheta) )

References:
- `PROJECT_GUIDELINES.md` (§3, §4.2, §4.6, §7).
- `Project Code/python_model_pseudocode.json` (MOD_05, MOD_06).
- Yang, Z., et al. (2022). Energy Conversion and Management, 267, 115942.
- Heywood, J. B. (2018). Internal Combustion Engine Fundamentals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp

from combustion import (
    CombustionParameters,
    resolve_combustion_parameters,
    wiebe_burned_fraction,
    wiebe_heat_release_rate_deg,
)
from config import FuelBlendConfig, OperatingConditions, RotaryGeometry, SimulationConfig
from kinematics import WankelKinematicsLookup
from submodels import (
    MechanicalLossBreakdown,
    compute_mechanical_losses,
    orifice_mass_flow_rate_deg,
    port_open_fraction,
    seal_leakage_rate_deg,
    woschni_heat_transfer_rate_deg,
)
from thermo_properties import MixtureThermoModel


@dataclass
class CycleResults:
    """Complete thermodynamic trajectories and extracted performance metrics."""

    rpm: float
    p_intake_pa: float
    theta_deg: np.ndarray
    volume_m3: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    mass_kg: np.ndarray
    area_m2: np.ndarray
    burned_fraction: np.ndarray
    dq_comb_ddeg: np.ndarray
    dq_wall_ddeg: np.ndarray
    dm_leak_ddeg: np.ndarray

    # Cycle summary metrics
    combustion_params: CombustionParameters
    m_trapped_ivc_kg: float
    m_fresh_ivc_kg: float
    m_fuel_kg: float
    residual_fraction: float
    volumetric_efficiency: float
    q_fuel_input_j: float
    q_comb_released_j: float
    q_wall_loss_j: float
    m_leak_total_kg: float
    h_leak_loss_j: float
    leak_mass_pct: float

    # Work & power metrics (single-rotor = 1 chamber cycle per shaft revolution)
    work_ind_net_j: float
    work_ind_gross_j: float
    work_pumping_j: float
    imep_net_pa: float
    imep_gross_pa: float
    p_ind_kw: float
    mechanical_losses: MechanicalLossBreakdown
    work_brake_j: float
    bmep_pa: float
    p_brake_kw: float
    torque_brake_nm: float
    eta_ith: float
    eta_bth: float
    bsfc_g_kwh: float
    exhaust_enthalpy_j: float
    p_max_pa: float
    t_max_k: float
    energy_balance_rel_error: float


class WankelCycleSolver:
    """Stiff ODE thermodynamic solver for a 4-stroke Wankel rotary chamber."""

    def __init__(self, sim_config: SimulationConfig) -> None:
        self.cfg = sim_config
        self.geom = sim_config.geometry
        self.fuel = sim_config.fuel
        self.kin = WankelKinematicsLookup(self.geom, grid_step_deg=0.25)
        self.thermo = MixtureThermoModel(self.fuel)

        # Effective port areas sized for a 0.22-0.26 dm^3 single-rotor Wankel
        # (~5.5 cm^2 intake, ~5.8 cm^2 exhaust with Cd=0.72).
        scale = self.geom.swept_volume / 2.411e-4
        self.a_intake_eff_max = 0.72 * 5.6e-4 * scale
        self.a_exhaust_eff_max = 0.74 * 6.0e-4 * scale

    def _integrate_pass(
        self,
        op: OperatingConditions,
        p0_pa: float,
        t0_k: float,
        residual_frac_est: float,
        comb_params: CombustionParameters,
        m_fuel_est: float,
        p_ivc_ref: float,
        t_ivc_ref: float,
        v_ivc_ref: float,
    ) -> tuple[np.ndarray, np.ndarray, CombustionParameters, float, float]:
        """Run one [0, 1080] deg ODE integration pass."""
        v0, _, _, _ = self.kin.evaluate(0.0)
        gas0 = self.thermo.evaluate(t0_k, burned_fraction=1.0, residual_fraction=1.0)
        m0 = p0_pa * v0 / (gas0.r_specific * t0_k)

        y0 = np.array([p0_pa, t0_k, m0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        theta_eval = np.arange(0.0, 1080.0 + 0.5, 0.5)

        gas_in = self.thermo.evaluate(op.t_intake, burned_fraction=0.0, residual_fraction=0.0)
        gas_exh_back = self.thermo.evaluate(750.0, burned_fraction=1.0, residual_fraction=1.0)
        v_ivc_geom, _, _, _ = self.kin.evaluate(self.geom.intake_close_deg)
        alt_method = "Radau" if self.cfg.ode_method == "BDF" else "BDF"

        def make_rhs(
            in_gas_ex_flag: bool,
            is_comb_flag: bool,
            m_fuel_val: float,
            p_ivc_val: float,
            t_ivc_val: float,
            v_ivc_val: float,
            comb_cfg: CombustionParameters,
        ):
            inv_v_ivc = 1.0 / v_ivc_val

            def rhs(theta: float, y: np.ndarray) -> np.ndarray:
                p = max(float(y[0]), 5000.0)
                t = max(float(y[1]), 200.0)
                m = max(float(y[2]), 1e-8)
                m_fresh = max(float(y[3]), 0.0)

                v, dv_ddeg, area, d_hyd = self.kin.evaluate(theta)

                xb = wiebe_burned_fraction(theta, comb_cfg) if m_fuel_val > 0.0 else 0.0
                res_frac = max(0.02, min(1.0, 1.0 - m_fresh / m))
                gas = self.thermo.evaluate(t, burned_fraction=xb, residual_fraction=res_frac)

                # 1. Port gas exchange flows
                in_frac = port_open_fraction(theta, self.geom.intake_open_deg, self.geom.intake_close_deg)
                ex_frac = port_open_fraction(theta, self.geom.exhaust_open_deg, self.geom.exhaust_close_deg)

                a_in = self.a_intake_eff_max * in_frac
                a_ex = self.a_exhaust_eff_max * ex_frac

                dm_in_forward = orifice_mass_flow_rate_deg(
                    op.p_intake, op.t_intake, p, a_in, gas_in.gamma, gas_in.r_specific, op.rpm
                )
                dm_in_back = orifice_mass_flow_rate_deg(
                    p, t, op.p_intake, a_in, gas.gamma, gas.r_specific, op.rpm
                )
                dm_ex_forward = orifice_mass_flow_rate_deg(
                    p, t, op.p_exhaust, a_ex, gas.gamma, gas.r_specific, op.rpm
                )
                dm_ex_back = orifice_mass_flow_rate_deg(
                    op.p_exhaust,
                    750.0,
                    p,
                    a_ex,
                    gas_exh_back.gamma,
                    gas_exh_back.r_specific,
                    op.rpm,
                )

                dm_in_net = dm_in_forward - dm_in_back
                dm_ex_net = dm_ex_forward - dm_ex_back

                # 2. Combustion heat release
                dq_comb = (
                    wiebe_heat_release_rate_deg(theta, m_fuel_val, self.fuel, comb_cfg)
                    if m_fuel_val > 0.0
                    else 0.0
                )

                # 3. Convective wall heat transfer (Modified Woschni)
                p_mot = p_ivc_val * ((v_ivc_val / v) ** 1.33)
                _ = inv_v_ivc
                _, dq_wall = woschni_heat_transfer_rate_deg(
                    theta_deg=theta,
                    pressure_pa=p,
                    temperature_k=t,
                    area_m2=area,
                    d_hyd_m=d_hyd,
                    rpm=op.rpm,
                    geom=self.geom,
                    p_ivc_pa=p_ivc_val,
                    t_ivc_k=t_ivc_val,
                    v_ivc_m3=v_ivc_val,
                    p_motored_pa=p_mot,
                    is_combustion_phase=is_comb_flag,
                    in_gas_exchange=in_gas_ex_flag,
                )

                # 4. Poiseuille-Couette seal leakage
                if m <= 1.0e-7:
                    dm_leak = 0.0
                else:
                    dm_leak = seal_leakage_rate_deg(
                        theta_deg=theta,
                        pressure_pa=p,
                        temperature_k=t,
                        rpm=op.rpm,
                        geom=self.geom,
                        gas=gas,
                        p_adjacent_pa=101325.0,
                    )

                # 5. Open-system energy & mass balance
                dm_ddeg = dm_in_net - dm_ex_net - dm_leak
                dm_fresh_ddeg = dm_in_forward - (m_fresh / m) * (dm_in_back + dm_ex_forward + dm_leak)

                u_chamber = gas.enthalpy - gas.r_specific * t
                h_in_term = (gas_in.enthalpy - u_chamber) * dm_in_forward - (
                    gas.r_specific * t
                ) * dm_in_back
                h_ex_term = -(gas.r_specific * t) * dm_ex_forward + (
                    gas_exh_back.enthalpy - u_chamber
                ) * dm_ex_back
                h_leak_term = -(gas.r_specific * t) * dm_leak

                dt_ddeg = (
                    dq_comb - dq_wall - p * dv_ddeg + h_in_term + h_ex_term + h_leak_term
                ) / max(m * gas.cv, 1e-9)

                dp_ddeg = p * ((dm_ddeg / m) + (dt_ddeg / t) - (dv_ddeg / v))

                return np.array(
                    [
                        dp_ddeg,
                        dt_ddeg,
                        dm_ddeg,
                        dm_fresh_ddeg,
                        dq_comb,
                        dq_wall,
                        dm_leak,
                        gas.enthalpy * dm_leak,
                    ],
                    dtype=float,
                )

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
                method=self.cfg.ode_method,
                t_eval=t_eval_arg,
                dense_output=need_dense,
                max_step=self.cfg.max_step_deg,
                rtol=self.cfg.rtol,
                atol=self.cfg.atol,
            )
            if not sol.success:
                sol = solve_ivp(
                    rhs_fn,
                    (t_start, t_end),
                    y_init,
                    method=alt_method,
                    t_eval=t_eval_arg,
                    dense_output=need_dense,
                    max_step=self.cfg.max_step_deg,
                    rtol=self.cfg.rtol,
                    atol=self.cfg.atol,
                )
            if not sol.success:
                raise RuntimeError(
                    f"Cycle ODE integration failed at {op.rpm} RPM [{t_start}, {t_end}]: {sol.message}"
                )
            y_end = sol.sol(t_end) if need_dense else sol.y[:, -1].copy()
            y_out = sol.y if len(t_eval_seg) > 0 else np.empty((len(y_init), 0), dtype=float)
            return y_out, y_end

        ivc_deg = self.geom.intake_close_deg
        spark_deg = comb_params.spark_timing_deg
        evo_deg = self.geom.exhaust_open_deg

        # Segment 1: Intake (0 -> IVC), gas-exchange Woschni C2 = 6.18
        mask1 = theta_eval <= ivc_deg
        y1_seg, y_ivc = integrate_segment(
            0.0,
            ivc_deg,
            y0,
            theta_eval[mask1],
            make_rhs(True, False, 0.0, p_ivc_ref, t_ivc_ref, v_ivc_ref, comb_params),
        )
        p_ivc = max(float(y_ivc[0]), 5000.0)
        t_ivc = max(float(y_ivc[1]), 200.0)
        m_fresh_ivc = max(float(y_ivc[3]), 0.0)
        m_fuel_locked = m_fresh_ivc / (1.0 + self.fuel.afr_actual) if m_fresh_ivc > 1e-9 else m_fuel_est

        # Segment 2: Compression prior to ignition (IVC -> spark_timing_deg)
        mask2 = (theta_eval > ivc_deg) & (theta_eval <= spark_deg)
        y2_seg, y_spk = integrate_segment(
            ivc_deg,
            spark_deg,
            y_ivc,
            theta_eval[mask2],
            make_rhs(False, False, 0.0, p_ivc, t_ivc, v_ivc_geom, comb_params),
        )
        p_spk = max(float(y_spk[0]), 5000.0)
        t_spk = max(float(y_spk[1]), 200.0)
        active_comb = resolve_combustion_parameters(
            rpm=op.rpm,
            spark_timing_deg=op.spark_timing_deg,
            t_spark_k=t_spk,
            p_spark_pa=p_spk,
            geom=self.geom,
            fuel=self.fuel,
        )

        # Segment 3: Combustion & closed expansion (spark_timing_deg -> EVO)
        mask3 = (theta_eval > spark_deg) & (theta_eval <= evo_deg)
        y3_seg, y_evo = integrate_segment(
            spark_deg,
            evo_deg,
            y_spk,
            theta_eval[mask3],
            make_rhs(False, True, m_fuel_locked, p_ivc, t_ivc, v_ivc_geom, active_comb),
        )

        # Segment 4: Exhaust blowdown & scavenging (EVO -> 1080 deg)
        mask4 = theta_eval > evo_deg
        y4_seg, _ = integrate_segment(
            evo_deg,
            1080.0,
            y_evo,
            theta_eval[mask4],
            make_rhs(True, False, m_fuel_locked, p_ivc, t_ivc, v_ivc_geom, active_comb),
        )

        y_full = np.hstack([y1_seg, y2_seg, y3_seg, y4_seg])
        return theta_eval, y_full, active_comb, m_fuel_locked, p_ivc

    def solve(self, op: OperatingConditions) -> CycleResults:
        """Solve a converged 1080-deg Wankel engine cycle at the given operating conditions."""
        # Estimate initial trapped fuel mass from intake density & swept volume
        gas_in = self.thermo.evaluate(op.t_intake, burned_fraction=0.0, residual_fraction=0.0)
        gas_exh_back = self.thermo.evaluate(750.0, burned_fraction=1.0, residual_fraction=1.0)
        rho_in = op.p_intake / (gas_in.r_specific * op.t_intake)
        v_ivc_est, _, _, _ = self.kin.evaluate(self.geom.intake_close_deg)
        m_fresh_est = 0.88 * rho_in * self.geom.swept_volume
        m_fuel_est = m_fresh_est / (1.0 + self.fuel.afr_actual)

        comb_init = resolve_combustion_parameters(
            rpm=op.rpm,
            spark_timing_deg=op.spark_timing_deg,
            t_spark_k=620.0,
            p_spark_pa=op.p_intake * (self.geom.compression_ratio ** 1.32),
            geom=self.geom,
            fuel=self.fuel,
        )

        # Pass 1: warm-up residual state & trapped mass
        theta1, y1, comb1, m_fuel1, p_ivc1 = self._integrate_pass(
            op=op,
            p0_pa=op.p_exhaust,
            t0_k=820.0,
            residual_frac_est=op.residual_fraction_init,
            comb_params=comb_init,
            m_fuel_est=m_fuel_est,
            p_ivc_ref=op.p_intake,
            t_ivc_ref=op.t_intake,
            v_ivc_ref=v_ivc_est,
        )

        # Pass 2: converged periodic boundary condition with smooth fixed combustion params
        p0_conv = float(y1[0, -1])
        t0_conv = float(np.clip(y1[1, -1], 500.0, 1400.0))
        idx_ivc = int(np.argmin(np.abs(theta1 - self.geom.intake_close_deg)))
        idx_spk = int(np.argmin(np.abs(theta1 - op.spark_timing_deg)))
        t_ivc1 = float(y1[1, idx_ivc])
        res_est = max(0.02, 1.0 - float(y1[3, idx_ivc]) / max(float(y1[2, idx_ivc]), 1e-8))
        comb2 = resolve_combustion_parameters(
            rpm=op.rpm,
            spark_timing_deg=op.spark_timing_deg,
            t_spark_k=float(y1[1, idx_spk]),
            p_spark_pa=float(y1[0, idx_spk]),
            geom=self.geom,
            fuel=self.fuel,
        )

        theta, y, comb, m_fuel_trapped, _ = self._integrate_pass(
            op=op,
            p0_pa=p0_conv,
            t0_k=t0_conv,
            residual_frac_est=res_est,
            comb_params=comb2,
            m_fuel_est=m_fuel1,
            p_ivc_ref=p_ivc1,
            t_ivc_ref=t_ivc1,
            v_ivc_ref=v_ivc_est,
        )

        p_arr = y[0]
        t_arr = y[1]
        m_arr = y[2]
        m_fresh_arr = y[3]
        q_comb_cum = y[4]
        q_wall_cum = y[5]
        m_leak_cum = y[6]
        h_leak_cum = y[7]

        v_arr = np.asarray(chamber_volume_array(theta, self.kin), dtype=float)
        dv_arr = np.asarray([self.kin.evaluate(th)[1] for th in theta], dtype=float)
        area_arr = np.asarray([self.kin.evaluate(th)[2] for th in theta], dtype=float)
        xb_arr = np.asarray([wiebe_burned_fraction(th, comb) for th in theta], dtype=float)

        dq_comb_arr = np.gradient(q_comb_cum, theta)
        dq_wall_arr = np.gradient(q_wall_cum, theta)
        dm_leak_arr = np.gradient(m_leak_cum, theta)

        # Extract IVC & EVO states
        idx_ivc = int(np.argmin(np.abs(theta - self.geom.intake_close_deg)))
        idx_evo = int(np.argmin(np.abs(theta - self.geom.exhaust_open_deg)))

        m_ivc = float(m_arr[idx_ivc])
        m_fresh_ivc = float(m_fresh_arr[idx_ivc])
        m_fuel = max(1e-10, m_fresh_ivc / (1.0 + self.fuel.afr_actual))
        residual_frac = max(0.0, 1.0 - m_fresh_ivc / max(m_ivc, 1e-9))

        # Volumetric efficiency relative to ambient air density (or manifold density if unthrottled)
        rho_amb = 101325.0 / (gas_in.r_specific * op.t_intake)
        m_air_trapped = m_fresh_ivc * (self.fuel.afr_actual / (1.0 + self.fuel.afr_actual))
        vol_eff = m_air_trapped / max(rho_amb * self.geom.swept_volume, 1e-9)

        # Closed-cycle energy balance sanity check between IVC and EVO
        gas_ivc = self.thermo.evaluate(float(t_arr[idx_ivc]), burned_fraction=0.0, residual_fraction=residual_frac)
        gas_evo = self.thermo.evaluate(float(t_arr[idx_evo]), burned_fraction=1.0, residual_fraction=residual_frac)
        u_ivc = float(m_arr[idx_ivc]) * (gas_ivc.enthalpy - gas_ivc.r_specific * float(t_arr[idx_ivc]))
        u_evo = float(m_arr[idx_evo]) * (gas_evo.enthalpy - gas_evo.r_specific * float(t_arr[idx_evo]))
        trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        w_closed = float(trapz_fn(p_arr[idx_ivc : idx_evo + 1], v_arr[idx_ivc : idx_evo + 1]))
        q_comb_closed = float(q_comb_cum[idx_evo] - q_comb_cum[idx_ivc])
        q_wall_closed = float(q_wall_cum[idx_evo] - q_wall_cum[idx_ivc])
        h_leak_closed = float(h_leak_cum[idx_evo] - h_leak_cum[idx_ivc])
        rhs_energy = q_comb_closed - q_wall_closed - w_closed - h_leak_closed
        delta_u = u_evo - u_ivc
        energy_err = abs(delta_u - rhs_energy) / max(q_comb_closed, 1.0)

        # Cyclic work integration: W_ind = integral(P dV) over [0, 1080] deg
        w_ind_net = float(trapz_fn(p_arr, v_arr))
        # High-pressure loop (270 to 810 deg) vs pumping loop (0-270 & 810-1080)
        mask_hp = (theta >= 270.0) & (theta <= 810.0)
        w_ind_gross = float(trapz_fn(p_arr[mask_hp], v_arr[mask_hp]))
        w_pump_signed = w_ind_net - w_ind_gross
        w_pumping_loss = max(0.0, -w_pump_signed)

        vd = self.geom.swept_volume
        imep_net = w_ind_net / vd
        imep_gross = w_ind_gross / vd

        # Single-rotor Wankel fires once per shaft revolution (3 chambers / 3 revs)
        cycles_per_sec = op.rpm / 60.0
        p_ind_kw = (w_ind_net * cycles_per_sec) / 1000.0

        p_max = float(np.max(p_arr))
        t_max = float(np.max(t_arr))

        mech = compute_mechanical_losses(op.rpm, p_max, self.geom, w_pumping_loss)
        w_brake = max(0.0, w_ind_net - mech.work_friction_total_j)
        bmep = w_brake / vd
        p_brake_kw = (w_brake * cycles_per_sec) / 1000.0
        omega_shaft = 2.0 * math.pi * op.rpm / 60.0
        torque_brake = (p_brake_kw * 1000.0) / max(omega_shaft, 1e-6)

        q_fuel_in = m_fuel * self.fuel.composite_lhv
        eta_ith = w_ind_net / max(q_fuel_in, 1e-9)
        eta_bth = w_brake / max(q_fuel_in, 1e-9)

        fuel_flow_g_hr = m_fuel * cycles_per_sec * 3600.0 * 1000.0
        bsfc = fuel_flow_g_hr / p_brake_kw if p_brake_kw > 0.05 else float("inf")

        q_wall_total = float(q_wall_cum[-1])
        m_leak_total = float(m_leak_cum[-1] - m_leak_cum[idx_ivc])
        leak_pct = 100.0 * max(0.0, m_leak_total) / max(m_ivc, 1e-9)
        h_leak_total = float(h_leak_cum[-1])

        # Exhaust enthalpy availability at EVO relative to 298.15 K
        exhaust_h = float(m_arr[idx_evo]) * gas_evo.enthalpy

        return CycleResults(
            rpm=op.rpm,
            p_intake_pa=op.p_intake,
            theta_deg=theta,
            volume_m3=v_arr,
            pressure_pa=p_arr,
            temperature_k=t_arr,
            mass_kg=m_arr,
            area_m2=area_arr,
            burned_fraction=xb_arr,
            dq_comb_ddeg=dq_comb_arr,
            dq_wall_ddeg=dq_wall_arr,
            dm_leak_ddeg=dm_leak_arr,
            combustion_params=comb,
            m_trapped_ivc_kg=m_ivc,
            m_fresh_ivc_kg=m_fresh_ivc,
            m_fuel_kg=m_fuel,
            residual_fraction=residual_frac,
            volumetric_efficiency=vol_eff,
            q_fuel_input_j=q_fuel_in,
            q_comb_released_j=float(q_comb_cum[-1]),
            q_wall_loss_j=q_wall_total,
            m_leak_total_kg=max(0.0, m_leak_total),
            h_leak_loss_j=h_leak_total,
            leak_mass_pct=leak_pct,
            work_ind_net_j=w_ind_net,
            work_ind_gross_j=w_ind_gross,
            work_pumping_j=w_pumping_loss,
            imep_net_pa=imep_net,
            imep_gross_pa=imep_gross,
            p_ind_kw=p_ind_kw,
            mechanical_losses=mech,
            work_brake_j=w_brake,
            bmep_pa=bmep,
            p_brake_kw=p_brake_kw,
            torque_brake_nm=torque_brake,
            eta_ith=eta_ith,
            eta_bth=eta_bth,
            bsfc_g_kwh=bsfc,
            exhaust_enthalpy_j=exhaust_h,
            p_max_pa=p_max,
            t_max_k=t_max,
            energy_balance_rel_error=energy_err,
        )


def chamber_volume_array(theta_arr: np.ndarray, kin: WankelKinematicsLookup) -> list[float]:
    """Helper returning exact chamber volume V(theta) across an angle array."""
    return [kin.evaluate(float(th))[0] for th in theta_arr]
