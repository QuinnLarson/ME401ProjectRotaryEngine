"""Wankel rotary engine epitrochoidal kinematics and surface area evaluator.

Implements:
1. Instantaneous chamber volume V(theta) and analytical derivative dV/dtheta
   over the complete 1080-deg eccentric shaft 4-stroke cycle.
2. Instantaneous exposed heat transfer surface area A(theta) combining:
   - Rotor flank surface area (with pocket recess enhancement)
   - Exposed epitrochoid stator housing arc area (exact analytical arc integral)
   - Front and rear flat side cover areas
3. Hydraulic diameter d_hyd(theta) = 4 * V(theta) / A(theta).
4. Interactive CLI input display allowing refinement of rotor radius (R),
   eccentricity (e), rotor width (b), and compression ratio (CR) with live
   geometric validation and tabular volume summary.

References:
- Yamamoto, K. (1981). Rotary Engine. Sankaido, Tokyo.
- Ansdale, R. F., & Lockley, D. (1968). The Wankel RC Engine.
- `Project Info & Goals/Heat Transfer Project Information.pdf` (§2.1, §2.2).
- `PROJECT_GUIDELINES.md` (§4.1).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import quad

from config import RotaryGeometry


def chamber_volume(theta_deg: float | np.ndarray, geom: RotaryGeometry) -> float | np.ndarray:
    """Compute instantaneous single-chamber volume V(theta) [m^3].

    Formula (`PROJECT_GUIDELINES.md` §4.1):
        V(theta) = V_min + (V_d / 2) * (1 - cos(2/3 * theta_rad))

    Cycle phase mapping (theta in [0, 1080] deg):
        - 0 deg:    TDC overlap (V = V_min)
        - 270 deg:  BDC end of intake (V = V_max)
        - 540 deg:  TDC compression / combustion (V = V_min)
        - 810 deg:  BDC end of expansion (V = V_max)
        - 1080 deg: TDC end of exhaust (V = V_min)
    """
    theta_rad = np.deg2rad(theta_deg)
    return geom.min_volume + 0.5 * geom.swept_volume * (1.0 - np.cos((2.0 / 3.0) * theta_rad))


def volume_derivative_rad(theta_deg: float | np.ndarray, geom: RotaryGeometry) -> float | np.ndarray:
    """Compute analytical derivative dV/d(theta_rad) [m^3/rad].

    Formula:
        dV/d(theta_rad) = (V_d / 3) * sin(2/3 * theta_rad)
    """
    theta_rad = np.deg2rad(theta_deg)
    return (geom.swept_volume / 3.0) * np.sin((2.0 / 3.0) * theta_rad)


def volume_derivative_deg(theta_deg: float | np.ndarray, geom: RotaryGeometry) -> float | np.ndarray:
    """Compute analytical derivative dV/d(theta_deg) [m^3/deg]."""
    return volume_derivative_rad(theta_deg, geom) * (math.pi / 180.0)


def epitrochoid_arc_length(psi_start: float, psi_end: float, r_gen: float, ecc: float) -> float:
    """Compute exact epitrochoid housing perimeter arc length between apex angles [m].

    Epitrochoid parametric curve:
        x(psi) = e*cos(3*psi) + R*cos(psi)
        y(psi) = e*sin(3*psi) + R*sin(psi)
        ds/dpsi = sqrt(9*e^2 + R^2 + 6*e*R*cos(2*psi))
    """
    integrand = lambda psi: math.sqrt(9.0 * ecc * ecc + r_gen * r_gen + 6.0 * ecc * r_gen * math.cos(2.0 * psi))
    arc_len, _ = quad(integrand, psi_start, psi_end, epsabs=1e-9, epsrel=1e-7)
    return arc_len


class WankelKinematicsLookup:
    """Precomputed high-resolution lookup tables for Wankel kinematics over [0, 1080] deg.

    Precomputing the epitrochoidal housing arc length and surface area on a 0.25-deg grid
    avoids calling quadrature inside the stiff ODE right-hand side while preserving
    exact epitrochoidal geometry accuracy.
    """

    def __init__(self, geom: RotaryGeometry, grid_step_deg: float = 0.25) -> None:
        self.geom = geom
        self.theta_grid = np.arange(0.0, 1080.0 + grid_step_deg * 0.5, grid_step_deg)
        self.volume_grid = chamber_volume(self.theta_grid, geom)
        self.dv_ddeg_grid = volume_derivative_deg(self.theta_grid, geom)
        self.dv_drad_grid = volume_derivative_rad(self.theta_grid, geom)

        # Rotor flank surface area: base convex flank arc (~1.06 * chord sqrt(3)*R) * width
        # plus internal surface area added by the recessed combustion pocket V_k.
        chord = math.sqrt(3.0) * geom.rotor_radius
        flank_arc_factor = 1.06
        pocket_area_factor = 1.0 + 0.28 * (geom.pocket_volume / max(geom.min_volume, 1e-9))
        self.a_rotor = flank_arc_factor * chord * geom.rotor_width * pocket_area_factor

        # Epitrochoid housing exposed area between bounding apexes:
        # At theta=0 deg, chamber is centered on minor axis (psi_mid = pi/2),
        # so apexes are at psi_0 = theta_rad/3 + pi/6 and psi_1 = psi_0 + 2*pi/3.
        a_housing = np.zeros_like(self.theta_grid)
        for idx, th_deg in enumerate(self.theta_grid):
            psi_0 = math.radians(th_deg) / 3.0 + math.pi / 6.0
            psi_1 = psi_0 + 2.0 * math.pi / 3.0
            arc = epitrochoid_arc_length(psi_0, psi_1, geom.rotor_radius, geom.eccentricity)
            a_housing[idx] = arc * geom.rotor_width

        self.a_housing_grid = a_housing
        # Two stationary flat side covers: cross-sectional area = (V(theta) - V_k) / b
        v_between_walls = np.maximum(self.volume_grid - geom.pocket_volume, 0.05 * geom.min_volume)
        self.a_side_grid = 2.0 * (v_between_walls / geom.rotor_width)
        self.a_total_grid = self.a_rotor + self.a_housing_grid + self.a_side_grid
        self.d_hyd_grid = 4.0 * self.volume_grid / np.maximum(self.a_total_grid, 1e-9)

    def evaluate(self, theta_deg: float) -> Tuple[float, float, float, float]:
        """Return (V [m^3], dV_ddeg [m^3/deg], A_total [m^2], d_hyd [m]) at theta_deg."""
        th = float(theta_deg) % 1080.0
        v = float(chamber_volume(th, self.geom))
        dv = float(volume_derivative_deg(th, self.geom))
        area = float(np.interp(th, self.theta_grid, self.a_total_grid))
        d_hyd = 4.0 * v / max(area, 1e-9)
        return v, dv, area, d_hyd


def summarize_geometry(geom: RotaryGeometry) -> Dict[str, float]:
    """Compute summary geometric metrics for display and sanity verification."""
    lookup = WankelKinematicsLookup(geom, grid_step_deg=1.0)
    a_v_tdc = float(lookup.a_total_grid[0] / lookup.volume_grid[0])
    a_v_bdc = float(lookup.a_total_grid[270] / lookup.volume_grid[270])
    return {
        "R_mm": geom.rotor_radius * 1e3,
        "e_mm": geom.eccentricity * 1e3,
        "b_mm": geom.rotor_width * 1e3,
        "K_shape_factor": geom.shape_factor,
        "V_d_cm3": geom.swept_volume * 1e6,
        "V_min_cm3": geom.min_volume * 1e6,
        "V_max_cm3": geom.max_volume * 1e6,
        "V_k_cm3": geom.pocket_volume * 1e6,
        "CR": geom.compression_ratio,
        "A_V_TDC_per_m": a_v_tdc,
        "A_V_BDC_per_m": a_v_bdc,
    }


def display_and_refine_geometry(geom: RotaryGeometry, interactive: bool = False) -> RotaryGeometry:
    """Display rotary chamber geometry and volume profile V(theta), optionally refining R, e, b, CR.

    Satisfies Prompt Objective 1:
    'Create an input display to refine rotary (Wankel) chamber volume as a function
     of shaft angle theta using rotor radius (R), eccentricity (e), and rotor width (b).'
    """
    current = RotaryGeometry(
        rotor_radius=geom.rotor_radius,
        eccentricity=geom.eccentricity,
        rotor_width=geom.rotor_width,
        compression_ratio=geom.compression_ratio,
        recess_type=geom.recess_type,
        wall_temperature=geom.wall_temperature,
        apex_seal_clearance=geom.apex_seal_clearance,
        side_seal_clearance=geom.side_seal_clearance,
        seal_land_length=geom.seal_land_length,
    )

    def _print_table(g: RotaryGeometry) -> None:
        stats = summarize_geometry(g)
        lookup = WankelKinematicsLookup(g, grid_step_deg=15.0)
        print("\n" + "=" * 72)
        print("WANKEL ROTARY CHAMBER GEOMETRY & V(theta) INPUT DISPLAY")
        print("=" * 72)
        print(
            f"  Rotor Radius (R)    : {stats['R_mm']:7.2f} mm   | "
            f"Shape Factor (K=R/e): {stats['K_shape_factor']:6.2f}"
        )
        print(
            f"  Eccentricity (e)    : {stats['e_mm']:7.2f} mm   | "
            f"Swept Volume (Vd)   : {stats['V_d_cm3']:6.2f} cm^3"
        )
        print(
            f"  Rotor Width (b)     : {stats['b_mm']:7.2f} mm   | "
            f"Clearance (Vmin)    : {stats['V_min_cm3']:6.2f} cm^3"
        )
        print(
            f"  Compression Ratio   : {stats['CR']:7.2f}      | "
            f"Max Volume (Vmax)   : {stats['V_max_cm3']:6.2f} cm^3"
        )
        print(
            f"  Recess Type         : {g.recess_type:>7s}      | "
            f"Pocket Vol (Vk)     : {stats['V_k_cm3']:6.2f} cm^3"
        )
        print(
            f"  Surface/Vol @ TDC   : {stats['A_V_TDC_per_m']:7.1f} 1/m  | "
            f"Surface/Vol @ BDC   : {stats['A_V_BDC_per_m']:6.1f} 1/m"
        )
        print("-" * 72)
        print("  Shaft Angle theta [deg]   Stroke        V(theta) [cm^3]   dV/dtheta [cm^3/deg]   A(theta) [cm^2]")
        key_angles = [
            (0.0, "Intake Start (TDC)"),
            (135.0, "Mid-Intake"),
            (270.0, "Intake End (BDC)"),
            (405.0, "Mid-Compression"),
            (540.0, "Ignition TDC"),
            (675.0, "Mid-Expansion"),
            (810.0, "Expansion End (BDC)"),
            (945.0, "Mid-Exhaust"),
            (1080.0, "Cycle End (TDC)"),
        ]
        for ang, label in key_angles:
            v, dv, area, _ = lookup.evaluate(ang)
            print(f"  {ang:8.1f}                  {label:<20s} {v*1e6:8.2f}          {dv*1e6:8.4f}             {area*1e4:7.2f}")
        print("=" * 72)

    _print_table(current)

    if interactive:
        while True:
            try:
                choice = input(
                    "\nRefine geometry parameters (R, e, b, CR, recess)? [y/N]: "
                ).strip().lower()
            except EOFError:
                break
            if choice not in {"y", "yes"}:
                break
            try:
                r_in = input(f"  Rotor radius R [mm] ({current.rotor_radius*1e3:.2f}): ").strip()
                e_in = input(f"  Eccentricity e [mm] ({current.eccentricity*1e3:.2f}): ").strip()
                b_in = input(f"  Rotor width b [mm]  ({current.rotor_width*1e3:.2f}): ").strip()
                cr_in = input(f"  Compression ratio   ({current.compression_ratio:.2f}): ").strip()
                rec_in = input(f"  Recess (LDR/SMR/TDR)({current.recess_type}): ").strip().upper()

                r_new = float(r_in) * 1e-3 if r_in else current.rotor_radius
                e_new = float(e_in) * 1e-3 if e_in else current.eccentricity
                b_new = float(b_in) * 1e-3 if b_in else current.rotor_width
                cr_new = float(cr_in) if cr_in else current.compression_ratio
                rec_new = rec_in if rec_in in {"LDR", "SMR", "TDR"} else current.recess_type

                if r_new <= 0 or e_new <= 0 or b_new <= 0 or r_new / e_new <= 3.5:
                    print("  [Invalid geometry] Ensure R, e, b > 0 and K = R/e > 3.5.")
                    continue
                if cr_new <= 1.5 or cr_new > 18.0:
                    print("  [Invalid CR] Ensure 1.5 < CR <= 18.0.")
                    continue

                current = RotaryGeometry(
                    rotor_radius=r_new,
                    eccentricity=e_new,
                    rotor_width=b_new,
                    compression_ratio=cr_new,
                    recess_type=rec_new,  # type: ignore[arg-type]
                    wall_temperature=current.wall_temperature,
                    apex_seal_clearance=current.apex_seal_clearance,
                    side_seal_clearance=current.side_seal_clearance,
                    seal_land_length=current.seal_land_length,
                )
                _print_table(current)
            except ValueError:
                print("  [Input Error] Please enter valid numeric values.")

    return current

