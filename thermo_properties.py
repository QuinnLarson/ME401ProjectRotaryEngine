"""Thermophysical gas property evaluator using NASA 7-coefficient polynomials.

Computes temperature- and composition-dependent thermodynamic and transport
properties for reacting gasoline (C8H18 surrogate) + hydrogen (H2) + air +
combustion products (CO2, H2O, N2, O2) mixtures across T in [250, 3500] K.

Uses exact analytical integration of the NASA Cp(T)/R polynomial for sensible
enthalpy relative to 298.15 K, giving machine-precision consistency with Cp(T)
and fast evaluation inside stiff ODE solvers.

Grounding & References:
- NASA Glenn / GRI-Mech 3.0 7-coefficient thermodynamic polynomials.
- Yang, Z., Du, Y., Geng, Q., & He, G. (2022). Energy Conversion and Management,
  267, 115942 (`Research/HybridRotaryPerformance.pdf`, §2.5.1 Type I real-gas effects).
- `PROJECT_GUIDELINES.md` (§3, §4.2, §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import FuelBlendConfig

R_UNIVERSAL = 8.314462618  # J / (mol * K)

SPECIES_ORDER = ("N2", "O2", "CO2", "H2O", "H2", "C8H18")

MOLAR_MASS: Dict[str, float] = {
    "N2": 0.0280134,
    "O2": 0.0319988,
    "CO2": 0.0440095,
    "H2O": 0.01801528,
    "H2": 0.00201588,
    "C8H18": 0.114232,
}

# NASA 7-term polynomials (a1..a5 for Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4)
# Format: (low_T_coeffs_200_1000K, high_T_coeffs_1000_3500K)
NASA_CP_COEFFS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "N2": (
        (3.298677, 1.408240e-3, -3.963222e-6, 5.641515e-9, -2.444854e-12),
        (2.926640, 1.487977e-3, -5.684760e-7, 1.009704e-10, -6.753351e-15),
    ),
    "O2": (
        (3.782456, -2.996734e-3, 9.847302e-6, -9.681295e-9, 3.243728e-12),
        (3.282538, 1.483088e-3, -7.579667e-7, 2.094706e-10, -2.167178e-14),
    ),
    "CO2": (
        (2.356773, 8.984597e-3, -7.123563e-6, 2.459190e-9, -1.436995e-13),
        (3.857460, 4.414370e-3, -2.214814e-6, 5.234902e-10, -4.720842e-14),
    ),
    "H2O": (
        (4.198641, -2.036434e-3, 6.520402e-6, -5.487971e-9, 1.771978e-12),
        (3.033992, 2.176918e-3, -1.640725e-7, -9.704199e-11, 1.682010e-14),
    ),
    "H2": (
        (2.344331, 7.980521e-3, -1.947815e-5, 2.015721e-8, -7.376118e-12),
        (3.337279, -4.940247e-5, 4.994568e-7, -1.795664e-10, 2.002554e-14),
    ),
    # Iso-octane vapor surrogate polynomial valid up to 1500 K (clamped above 1500 K)
    "C8H18": (
        (-0.55313, 0.09686, -5.853e-5, 1.564e-8, -1.49e-12),
        (-0.55313, 0.09686, -5.853e-5, 1.564e-8, -1.49e-12),
    ),
}

_T_REF = 298.15
_T_SPLIT = 1000.0
_LOW_COEFFS = np.array([NASA_CP_COEFFS[s][0] for s in SPECIES_ORDER], dtype=float)   # (6, 5)
_HIGH_COEFFS = np.array([NASA_CP_COEFFS[s][1] for s in SPECIES_ORDER], dtype=float)  # (6, 5)
_MOLAR_MASSES_ARR = np.array([MOLAR_MASS[s] for s in SPECIES_ORDER], dtype=float)


def _antiderivative_h_over_r(coeffs: np.ndarray, t: float) -> np.ndarray:
    """Analytical antiderivative of (Cp/R) dT for all 6 species at temperature T."""
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    t_vec = np.array([t, 0.5 * t2, t3 / 3.0, 0.25 * t4, 0.2 * t5], dtype=float)
    return coeffs @ t_vec


_H_LOW_AT_REF = _antiderivative_h_over_r(_LOW_COEFFS, _T_REF)
_H_LOW_AT_SPLIT = _antiderivative_h_over_r(_LOW_COEFFS, _T_SPLIT)
_H_HIGH_AT_SPLIT = _antiderivative_h_over_r(_HIGH_COEFFS, _T_SPLIT)
_H_C8_AT_1500 = float(
    (_H_LOW_AT_SPLIT[5] - _H_LOW_AT_REF[5])
    + (_antiderivative_h_over_r(_HIGH_COEFFS, 1500.0)[5] - _H_HIGH_AT_SPLIT[5])
)
_CP_C8_AT_1500 = float(
    _LOW_COEFFS[5] @ np.array([1.0, 1500.0, 1500.0**2, 1500.0**3, 1500.0**4], dtype=float)
)


@dataclass(frozen=True)
class GasProperties:
    """Instantaneous thermodynamic and transport properties of the working fluid."""

    cp: float  # J / (kg * K)
    cv: float  # J / (kg * K)
    gamma: float  # -
    r_specific: float  # J / (kg * K)
    molar_mass: float  # kg / mol
    enthalpy: float  # J / kg (sensible enthalpy relative to 298.15 K)
    viscosity: float  # Pa * s
    conductivity: float  # W / (m * K)
    prandtl: float  # -


class MixtureThermoModel:
    """Fast real-gas mixture evaluator for gasoline-hydrogen-air-exhaust mixtures."""

    def __init__(self, fuel: FuelBlendConfig) -> None:
        self.fuel = fuel
        n_c8 = (1.0 - fuel.h2_mass_fraction) / MOLAR_MASS["C8H18"]
        n_h2 = fuel.h2_mass_fraction / MOLAR_MASS["H2"]
        n_o2_stoich = 12.5 * n_c8 + 0.5 * n_h2
        n_o2_in = fuel.excess_air_ratio * n_o2_stoich
        n_n2 = 3.76 * n_o2_in

        self._moles_unburned = np.array([n_n2, n_o2_in, 0.0, 0.0, n_h2, n_c8], dtype=float)
        self._moles_burned = np.array(
            [
                n_n2,
                max(0.0, (fuel.excess_air_ratio - 1.0) * n_o2_stoich),
                8.0 * n_c8,
                9.0 * n_c8 + n_h2,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self._air_moles = np.array([0.79, 0.21, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def evaluate(
        self,
        temp_k: float,
        burned_fraction: float = 0.0,
        residual_fraction: float = 0.05,
        pure_air_mode: bool = False,
    ) -> GasProperties:
        """Evaluate mixture thermodynamic & transport properties at (T, x_b, x_res)."""
        t = 220.0 if temp_k < 220.0 else (3500.0 if temp_k > 3500.0 else float(temp_k))

        if pure_air_mode:
            mole_fracs = self._air_moles
        else:
            xb_eff = residual_fraction + (1.0 - residual_fraction) * burned_fraction
            if xb_eff < 0.0:
                xb_eff = 0.0
            elif xb_eff > 1.0:
                xb_eff = 1.0
            moles = (1.0 - xb_eff) * self._moles_unburned + xb_eff * self._moles_burned
            mole_fracs = moles / moles.sum()

        m_mix = float(mole_fracs @ _MOLAR_MASSES_ARR)
        r_spec = R_UNIVERSAL / m_mix

        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t_pow = np.array([1.0, t, t2, t3, t4], dtype=float)

        if t <= _T_SPLIT:
            cp_over_r = _LOW_COEFFS @ t_pow
            h_over_r = _antiderivative_h_over_r(_LOW_COEFFS, t) - _H_LOW_AT_REF
        else:
            cp_over_r = _HIGH_COEFFS @ t_pow
            h_over_r = (_H_LOW_AT_SPLIT - _H_LOW_AT_REF) + (
                _antiderivative_h_over_r(_HIGH_COEFFS, t) - _H_HIGH_AT_SPLIT
            )
            if t > 1500.0:
                cp_over_r[5] = _CP_C8_AT_1500
                h_over_r[5] = _H_C8_AT_1500 + _CP_C8_AT_1500 * (t - 1500.0)

        cp_mass = R_UNIVERSAL * float(mole_fracs @ cp_over_r) / m_mix
        cv_mass = cp_mass - r_spec
        gamma = cp_mass / (cv_mass if cv_mass > 1e-6 else 1e-6)
        h_mass = R_UNIVERSAL * float(mole_fracs @ h_over_r) / m_mix

        # Dynamic viscosity via Sutherland's law for air/combustion gas [Pa*s]
        mu = 1.716e-5 * ((t / 273.15) ** 1.5) * (383.55 / (t + 110.4))
        # Modified Eucken correlation for gas thermal conductivity k [W/(m*K)]
        k_cond = mu * (cp_mass + 1.25 * r_spec)
        prandtl = (mu * cp_mass) / (k_cond if k_cond > 1e-9 else 1e-9)

        return GasProperties(
            cp=cp_mass,
            cv=cv_mass,
            gamma=gamma,
            r_specific=r_spec,
            molar_mass=m_mix,
            enthalpy=h_mass,
            viscosity=mu,
            conductivity=k_cond,
            prandtl=prandtl,
        )
