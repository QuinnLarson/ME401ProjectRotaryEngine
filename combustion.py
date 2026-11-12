"""Combustion heat release, flame speed, and hydrogen blending submodels.

Implements:
1. Metghalchi & Keck (1982) laminar flame speed correlations for gasoline and
   hydrogen with temperature/pressure power-law scaling (Yang et al. 2022, Eq. 10-12).
2. Logarithmic energy-weighted mixture laminar flame speed S_mix (Yang et al. 2022, Eq. 13).
3. Turbulent flame speed enhancement S_T = S_mix + f(u', recess) and combustion
   angular duration scaling across RPM, hydrogen fraction x_H2, and recess geometry
   (LDR vs. SMR vs. TDR per Boughou & Mohiuddin 2020 and Meng et al. 2023).
4. Single Wiebe mass-fraction burned x_b(theta) and chemical heat release rate dQ_comb/dtheta.

References:
- Yang, Z., Du, Y., Geng, Q., & He, G. (2022). Energy Conversion and Management,
  267, 115942 (`Research/HybridRotaryPerformance.pdf`, §2.4, Fig. 3, Eq. 8-14).
- Metghalchi, M., & Keck, J. C. (1982). Combustion and Flame, 48, 191-210.
- Boughou, S., & Mohiuddin, A. K. M. (2020). (`Research/CombustionChamberDesignEffect-RotaryEngine.pdf`).
- Meng, H., et al. (2023). (`Research/WankelRotartyEngine.pdf`, §2.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import FuelBlendConfig, RecessType, RotaryGeometry


@dataclass(frozen=True)
class CombustionParameters:
    """Resolved combustion timing, duration, and Wiebe parameters for a cycle."""

    spark_timing_deg: float
    duration_deg: float
    wiebe_a: float
    wiebe_m: float
    combustion_efficiency: float
    laminar_flame_speed_mps: float
    turbulent_flame_speed_mps: float


def reference_laminar_flame_speed(fuel: FuelBlendConfig) -> tuple[float, float]:
    """Return (S_L_gasoline_ref, S_L_h2_ref) [m/s] at T_ref=298 K, P_ref=1 bar.

    Grounded in Metghalchi & Keck (1982) and Fig. 3 of Yang et al. (2022):
    - Gasoline peaks ~0.32 m/s near equivalence ratio phi = 1.08, dropping lean.
    - Hydrogen peaks ~2.95 m/s near phi = 1.7, ~2.10 m/s at stoichiometric phi = 1.0.
    """
    phi_eq = 1.0 / max(fuel.excess_air_ratio, 0.5)
    # Gasoline polynomial fit matching Fig. 3 (Yang et al. 2022)
    s_g_ref = max(0.06, 0.32 - 0.85 * (phi_eq - 1.08) ** 2)
    # Hydrogen fit matching Fig. 3 (Yang et al. 2022) for phi in [0.4, 1.6]
    s_h_ref = max(0.25, 2.95 - 1.45 * (1.58 - min(phi_eq, 1.58)) ** 2)
    return s_g_ref, s_h_ref


def compute_blended_flame_speed(
    temp_k: float,
    pressure_pa: float,
    fuel: FuelBlendConfig,
) -> float:
    """Compute instantaneous laminar flame speed S_mix(T, P, lambda, x_H2) [m/s].

    Equations (10)-(13) of Yang et al. (2022):
        omega = 2.18 - 0.8 * (1 / lambda)
        upsilon = -0.16 + 0.22 * (1 / lambda - 1)
        S_i = S_{i,ref} * (T / 298)^omega * (P / 1e5)^upsilon
        S_mix = exp( (Q_g/Q_tot)*ln(S_g) + (Q_h/Q_tot)*ln(S_h) )
    """
    t_ref = 298.0
    p_ref = 1.0e5
    inv_lam = 1.0 / max(fuel.excess_air_ratio, 0.5)

    omega_exp = 2.18 - 0.8 * inv_lam
    upsilon_exp = -0.16 + 0.22 * (inv_lam - 1.0)

    t_ratio = max(temp_k, 250.0) / t_ref
    p_ratio = max(pressure_pa, 2.0e4) / p_ref

    s_g_ref, s_h_ref = reference_laminar_flame_speed(fuel)
    s_g = s_g_ref * (t_ratio ** omega_exp) * (p_ratio ** upsilon_exp)
    s_h = s_h_ref * (t_ratio ** omega_exp) * (p_ratio ** upsilon_exp)

    f_h2_energy = fuel.h2_energy_fraction
    f_g_energy = 1.0 - f_h2_energy
    s_mix = math.exp(f_g_energy * math.log(max(s_g, 1e-4)) + f_h2_energy * math.log(max(s_h, 1e-4)))
    return s_mix


def resolve_combustion_parameters(
    rpm: float,
    spark_timing_deg: float,
    t_spark_k: float,
    p_spark_pa: float,
    geom: RotaryGeometry,
    fuel: FuelBlendConfig,
) -> CombustionParameters:
    """Resolve Wiebe duration, form factor, and combustion efficiency for a cycle.

    Accounts for:
    - Laminar flame speed S_mix(T_spark, P_spark, lambda, x_H2)
    - Rotor swirl/turbulence intensity u' ~ v_rotor (Heat Transfer Project Info §4.2)
    - Recess pocket geometry (Boughou & Mohiuddin 2020; Meng et al. 2023):
      * LDR (Leading Deep Recess): higher swirl near leading apex (+12% flame speed,
        faster early burn m_w=1.85, improved completeness).
      * SMR (Central Symmetric Recess): balanced baseline (m_w=2.05).
      * TDR (Trailing Deep Recess): slower early burn (m_w=2.25) but burns trailing
        end-gas slightly better than baseline SMR.
    - Wankel wall quenching mitigation by hydrogen enrichment (Meng et al. 2023 §2.5):
      short quenching distance of H2 raises combustion efficiency under lean conditions.
    """
    recess_effects: dict[RecessType, tuple[float, float, float]] = {
        # (turbulence_swirl_mult, wiebe_m, base_comb_eff_bonus)
        "LDR": (1.12, 1.85, 0.015),
        "SMR": (1.00, 2.05, 0.000),
        "TDR": (0.94, 2.25, 0.008),
    }
    swirl_mult, wiebe_m, eff_bonus = recess_effects[geom.recess_type]

    # Reference pure-gasoline laminar flame speed at typical spark TDC state (650 K, 15 bar)
    ref_fuel = FuelBlendConfig(h2_mass_fraction=0.0, excess_air_ratio=1.0)
    s_l_ref = compute_blended_flame_speed(650.0, 1.5e6, ref_fuel)
    s_l_actual = compute_blended_flame_speed(t_spark_k, p_spark_pa, fuel)

    # Rotor tip/face characteristic velocity and turbulence contribution u'
    omega_shaft = 2.0 * math.pi * rpm / 60.0
    v_rotor_ref = geom.rotor_radius * (2.0 * math.pi * 5000.0 / 60.0) / 3.0
    v_rotor = geom.rotor_radius * omega_shaft / 3.0

    # Turbulent flame speed S_T = S_L + k_turb * v_rotor * swirl_mult
    k_turb = 0.36
    s_t_ref = s_l_ref + k_turb * v_rotor_ref
    s_t_actual = s_l_actual + k_turb * v_rotor * swirl_mult

    # Nominal 10-90% / 0-99% Wiebe burn duration in eccentric shaft degrees at 5000 RPM:
    # ~72 deg EA (equivalent to ~48 deg crank angle in a reciprocating engine).
    delta_phi_ref = 72.0
    duration_deg = delta_phi_ref * (rpm / 5000.0) * (s_t_ref / max(s_t_actual, 0.2))
    duration_deg = max(32.0, min(115.0, duration_deg))

    # Combustion completeness (accounting for Wankel trailing-edge quenching & lean burn)
    # Pure gasoline Wankel typically achieves ~93-95%; H2 addition cuts unburned HC.
    lean_penalty = 0.035 * max(0.0, fuel.excess_air_ratio - 1.10) ** 1.5
    h2_recovery = 0.045 * (1.0 - math.exp(-8.0 * fuel.h2_mass_fraction))
    eta_comb = min(0.99, max(0.86, 0.945 + eff_bonus + h2_recovery - lean_penalty))

    return CombustionParameters(
        spark_timing_deg=spark_timing_deg,
        duration_deg=duration_deg,
        wiebe_a=5.0,
        wiebe_m=wiebe_m,
        combustion_efficiency=eta_comb,
        laminar_flame_speed_mps=s_l_actual,
        turbulent_flame_speed_mps=s_t_actual,
    )


def wiebe_burned_fraction(theta_deg: float, params: CombustionParameters) -> float:
    """Evaluate cumulative mass fraction burned x_b(theta) in [0, 1]."""
    if theta_deg <= params.spark_timing_deg:
        return 0.0
    tau = (theta_deg - params.spark_timing_deg) / params.duration_deg
    if tau >= 1.4:
        return 1.0
    return 1.0 - math.exp(-params.wiebe_a * (tau ** (params.wiebe_m + 1.0)))


def wiebe_heat_release_rate_deg(
    theta_deg: float,
    m_fuel_trapped: float,
    fuel: FuelBlendConfig,
    params: CombustionParameters,
) -> float:
    """Evaluate chemical heat release rate dQ_comb/d(theta_deg) [J/deg].

    Formula:
        dQ_comb/dtheta = m_fuel * eta_comb * LHV_mix * (dx_b / dtheta)
    """
    if theta_deg <= params.spark_timing_deg:
        return 0.0
    tau = (theta_deg - params.spark_timing_deg) / params.duration_deg
    if tau >= 1.4:
        return 0.0
    a = params.wiebe_a
    m = params.wiebe_m
    dxb_dtheta = (
        a
        * (m + 1.0)
        / params.duration_deg
        * (tau ** m)
        * math.exp(-a * (tau ** (m + 1.0)))
    )
    return m_fuel_trapped * params.combustion_efficiency * fuel.composite_lhv * dxb_dtheta

