"""Physical submodels for convective heat transfer, seal blow-by, port flow, and friction.

Implements:
1. Modified Woschni convective heat transfer correlation adapted for epitrochoidal
   rotary kinematics (Woschni 1968; Yang et al. 2022; `PROJECT_GUIDELINES.md` §4.4).
2. Combined Poiseuille (pressure-driven) + Couette (shear-driven) gas leakage across
   apex and side seals (Yang et al. 2022 Eq. 20-21; Jeng et al. 2013; Meng et al. 2023).
3. Quasi-steady compressible orifice flow for intake and exhaust ports over the
   1080-deg eccentric shaft cycle.
4. Mechanical friction decomposition (bearing/gear rubbing friction + apex/side seal
   sliding drag) calibrated to Wankel rotary FMEP literature (Rachel & Schock 1991;
   Meng et al. 2023 §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from config import RotaryGeometry
from thermo_properties import GasProperties


@dataclass(frozen=True)
class MechanicalLossBreakdown:
    """Breakdown of mechanical friction and pumping losses for one cycle."""

    fmep_rubbing_pa: float  # Bearing, eccentric journal, gear, oil churning FMEP [Pa]
    fmep_seal_drag_pa: float  # Apex & side seal sliding friction FMEP [Pa]
    fmep_total_pa: float  # Total mechanical FMEP [Pa]
    work_rubbing_j: float  # J per chamber cycle
    work_seal_drag_j: float  # J per chamber cycle
    work_friction_total_j: float  # J per chamber cycle
    pmep_pa: float  # Pumping mean effective pressure (from intake/exhaust P-V loop) [Pa]
    work_pumping_j: float  # Pumping work magnitude (>= 0 when negative loop) [J]


def woschni_heat_transfer_rate_deg(
    theta_deg: float,
    pressure_pa: float,
    temperature_k: float,
    area_m2: float,
    d_hyd_m: float,
    rpm: float,
    geom: RotaryGeometry,
    p_ivc_pa: float,
    t_ivc_k: float,
    v_ivc_m3: float,
    p_motored_pa: float,
    is_combustion_phase: bool,
    in_gas_exchange: bool | None = None,
) -> tuple[float, float]:
    """Evaluate modified Woschni h_c [W/(m^2*K)] and dQ_wall/d(theta_deg) [J/deg].

    Standard Woschni (1968) SI coefficient with P in kPa:
        h_c = 3.26 * d_hyd^(-0.2) * P_kPa^(0.8) * T^(-0.53) * w^(0.8)  [W/(m^2*K)]

    Adapted for Wankel rotor velocity (`PROJECT_GUIDELINES.md` §4.4):
        v_rotor = R * omega_rotor = R * (omega_shaft / 3)
        w = C_2 * v_rotor + C_3 * (V_d * T_ivc / (P_ivc * V_ivc)) * (P - P_mot)

    where C_2 = 2.28 (closed cycle) or 6.18 (gas exchange), and C_3 = 0.00324 during
    combustion/expansion (Heywood 2018; Yang et al. 2022).

    Returns:
        (h_c [W/(m^2*K)], dQ_wall_ddeg [J/deg], positive when heat flows gas -> wall).
    """
    omega_shaft = 2.0 * math.pi * rpm / 60.0
    omega_rotor = omega_shaft / 3.0
    v_rotor = geom.rotor_radius * omega_rotor

    # Gas exchange (intake 0-285, exhaust 795-1080) vs closed compression/expansion
    if in_gas_exchange is None:
        in_gas_exchange = (theta_deg < geom.intake_close_deg) or (theta_deg > geom.exhaust_open_deg)
    c2 = 6.18 if in_gas_exchange else 2.28
    c3 = 0.00324 if is_combustion_phase else 0.0

    comb_vel = c3 * (geom.swept_volume * t_ivc_k / max(p_ivc_pa * v_ivc_m3, 1e-6)) * max(
        0.0, pressure_pa - p_motored_pa
    )
    w_vel = max(0.5, c2 * v_rotor + comb_vel)

    p_kpa = max(pressure_pa, 1000.0) / 1000.0
    t_safe = max(temperature_k, 220.0)
    d_safe = max(d_hyd_m, 0.002)

    # C1 = 3.26 for P in kPa (equivalent to 129.8 for P in bar)
    h_c = 3.26 * (d_safe ** -0.2) * (p_kpa ** 0.8) * (t_safe ** -0.53) * (w_vel ** 0.8)

    # Convert dQ/dt [W] -> dQ/dtheta_deg [J/deg] using dtheta_deg/dt = 6 * RPM
    deg_per_sec = 6.0 * rpm
    dq_wall_ddeg = h_c * area_m2 * (t_safe - geom.wall_temperature) / deg_per_sec
    return h_c, dq_wall_ddeg


def seal_leakage_rate_deg(
    theta_deg: float,
    pressure_pa: float,
    temperature_k: float,
    rpm: float,
    geom: RotaryGeometry,
    gas: GasProperties,
    p_adjacent_pa: float = 101325.0,
) -> float:
    """Evaluate Poiseuille-Couette blow-by mass loss rate dm_leak/d(theta_deg) [kg/deg].

    Equations (20)-(21) of Yang et al. (2022) and `PROJECT_GUIDELINES.md` §4.5:
    - Apex seals (2 bounding seals, total width = 2 * b):
        q_apex = (w_apex * delta_apex^3 / (12 * mu * L)) * dP + (w_apex * delta_apex * v_wall) / 2
    - Side seals (2 side faces, total perimeter ~ 2 * sqrt(3) * R):
        q_side = (w_side * delta_side^3 / (12 * mu * L)) * dP  (Couette ignored radially)

    Positive return value indicates mass escaping the working chamber.
    """
    # Blow-by is negligible while ports are wide open near atmospheric pressure
    if theta_deg < geom.intake_close_deg or theta_deg > geom.exhaust_open_deg:
        delta_p = max(0.0, pressure_pa - p_adjacent_pa)
        if delta_p < 5000.0:
            return 0.0
    else:
        delta_p = max(0.0, pressure_pa - p_adjacent_pa)

    rho = pressure_pa / max(gas.r_specific * temperature_k, 1e-3)
    mu = max(gas.viscosity, 1e-6)
    l_land = geom.seal_land_length

    # 2 apex seals bounding the chamber
    w_apex = 2.0 * geom.rotor_width
    d_apex = geom.apex_seal_clearance
    omega_shaft = 2.0 * math.pi * rpm / 60.0
    v_wall = geom.rotor_radius * (omega_shaft / 3.0)

    q_apex_poiseuille = (w_apex * (d_apex ** 3) * delta_p) / (12.0 * mu * l_land)
    q_apex_couette = 0.5 * w_apex * d_apex * v_wall if delta_p > 1000.0 else 0.0

    # 2 side seal arcs (chord length ~ sqrt(3)*R per face)
    w_side = 2.0 * math.sqrt(3.0) * geom.rotor_radius
    d_side = geom.side_seal_clearance
    q_side_poiseuille = (w_side * (d_side ** 3) * delta_p) / (12.0 * mu * l_land)

    q_total = q_apex_poiseuille + q_apex_couette + q_side_poiseuille
    deg_per_sec = 6.0 * rpm
    return (rho * q_total) / deg_per_sec


def port_open_fraction(theta_deg: float, open_deg: float, close_deg: float, ramp_deg: float = 18.0) -> float:
    """Smooth raised-cosine port area profile in [0, 1] for ODE numerical smoothness."""
    if theta_deg <= open_deg or theta_deg >= close_deg:
        return 0.0
    if theta_deg < open_deg + ramp_deg:
        s = (theta_deg - open_deg) / ramp_deg
        return 0.5 * (1.0 - math.cos(math.pi * s))
    if theta_deg > close_deg - ramp_deg:
        s = (close_deg - theta_deg) / ramp_deg
        return 0.5 * (1.0 - math.cos(math.pi * s))
    return 1.0


def orifice_mass_flow_rate_deg(
    p_up_pa: float,
    t_up_k: float,
    p_down_pa: float,
    area_eff_m2: float,
    gamma: float,
    r_spec: float,
    rpm: float,
) -> float:
    """Isentropic compressible orifice flow rate dm/d(theta_deg) [kg/deg] (Heywood 2018)."""
    if area_eff_m2 <= 1e-12 or p_up_pa <= p_down_pa:
        return 0.0

    pr = p_down_pa / p_up_pa
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if pr <= pr_crit:
        # Choked flow
        flow_func = math.sqrt(gamma / (r_spec * t_up_k)) * (
            (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
    else:
        term = max(0.0, (pr ** (2.0 / gamma)) - (pr ** ((gamma + 1.0) / gamma)))
        flow_func = math.sqrt((2.0 * gamma) / (r_spec * t_up_k * (gamma - 1.0)) * term)

    mdot_kg_s = area_eff_m2 * p_up_pa * flow_func
    return mdot_kg_s / (6.0 * rpm)


def compute_mechanical_losses(
    rpm: float,
    p_max_pa: float,
    geom: RotaryGeometry,
    work_pumping_j: float,
) -> MechanicalLossBreakdown:
    """Calculate mechanical rubbing friction, seal drag, and pumping FMEP/work.

    Grounded in Wankel friction literature (Rachel & Schock 1991; Meng et al. 2023 §2.1):
    - Bearing/eccentric/gear rubbing FMEP scales modestly with P_max and linearly/quadratically
      with shaft speed.
    - Apex and side seal drag scales with gas pressure loading and sliding velocity
      (v_rotor = R * omega_shaft / 3).

    Calibrated to typical small single-rotor Wankel FMEP:
    ~0.85 bar at 2000 RPM, ~1.45 bar at 5000 RPM, ~2.15 bar at 7000 RPM.
    """
    omega_shaft = 2.0 * math.pi * rpm / 60.0
    v_apex = geom.rotor_radius * (omega_shaft / 3.0)
    n_krpm = rpm / 1000.0
    p_max_bar = p_max_pa / 1.0e5

    # Bearing, internal gear, and oil churning rubbing friction [Pa]
    fmep_rubbing_bar = 0.28 + 0.0045 * p_max_bar + 0.052 * n_krpm + 0.011 * (n_krpm ** 2)
    # Apex + side seal sliding friction [Pa] (exponential/quadratic with speed + gas load)
    fmep_seal_bar = 0.24 + 0.0065 * p_max_bar + 0.018 * v_apex + 0.00055 * (v_apex ** 2)

    fmep_rubbing_pa = fmep_rubbing_bar * 1.0e5
    fmep_seal_pa = fmep_seal_bar * 1.0e5
    fmep_total_pa = fmep_rubbing_pa + fmep_seal_pa

    vd = geom.swept_volume
    w_rub = fmep_rubbing_pa * vd
    w_seal = fmep_seal_pa * vd
    w_fric = fmep_total_pa * vd
    pmep_pa = work_pumping_j / vd

    return MechanicalLossBreakdown(
        fmep_rubbing_pa=fmep_rubbing_pa,
        fmep_seal_drag_pa=fmep_seal_pa,
        fmep_total_pa=fmep_total_pa,
        work_rubbing_j=w_rub,
        work_seal_drag_j=w_seal,
        work_friction_total_j=w_fric,
        pmep_pa=pmep_pa,
        work_pumping_j=work_pumping_j,
    )

