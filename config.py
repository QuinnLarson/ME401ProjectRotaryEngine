"""Configuration dataclasses and physical constants for the ME401 rotary simulation.

Grounding & References:
- Baseline Wankel geometry and operating targets:
  * Quinn Larson, `PROJECT_GUIDELINES.md` (Section 2, Section 4) and
    `Project Code/python_model_pseudocode.json` (MOD_01).
  * Yang, Z., Du, Y., Geng, Q., & He, G. (2022). Energy Conversion and Management,
    267, 115942 (`Research/HybridRotaryPerformance.pdf`, Table 1).
  * Boughou, S., & Mohiuddin, A. K. M. (2020). Int. J. Automotive Engineering,
    11(4), 200-212 (`Research/CombustionChamberDesignEffect-RotaryEngine.pdf`).
  * Meng, H., Ji, C., Wang, S., & Yang, J. (2023). Fuel, 335, 127043
    (`Research/WankelRotartyEngine.pdf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

RecessType = Literal["LDR", "SMR", "TDR"]


@dataclass
class RotaryGeometry:
    """Epitrochoidal Wankel rotary engine geometry (single-rotor, 3 chambers).

    All lengths are in meters [m] and volumes in cubic meters [m^3].

    Attributes:
        rotor_radius: Generating radius R [m] (default 0.089 m = 89 mm).
        eccentricity: Eccentric shaft offset e [m] (default 0.013 m = 13 mm).
        rotor_width: Rotor / housing axial thickness b [m] (default 0.0401 m = 40.1 mm).
        compression_ratio: Target geometric compression ratio CR = V_max / V_min [-].
            Literature sweep values from Boughou & Mohiuddin (2020) / Jeng et al. (2013):
            8.33, 9.55, 10.18.
        recess_type: Rotor combustion pocket placement ('LDR', 'SMR', or 'TDR').
        wall_temperature: Effective time-averaged chamber wall temperature [K].
        apex_seal_clearance: Effective apex seal clearance gap height delta [m].
        side_seal_clearance: Effective side seal clearance gap height delta [m].
        seal_land_length: Effective sealing land thickness L [m].
        intake_open_deg: Eccentric shaft angle [deg] when intake port opens.
        intake_close_deg: Eccentric shaft angle [deg] when intake port closes.
        exhaust_open_deg: Eccentric shaft angle [deg] when exhaust port opens.
        exhaust_close_deg: Eccentric shaft angle [deg] when exhaust port closes.
    """

    rotor_radius: float = 0.089
    eccentricity: float = 0.013
    rotor_width: float = 0.0401
    compression_ratio: float = 9.55
    recess_type: RecessType = "LDR"
    wall_temperature: float = 430.0
    apex_seal_clearance: float = 22.0e-6
    side_seal_clearance: float = 12.0e-6
    seal_land_length: float = 0.004
    intake_open_deg: float = 0.0
    intake_close_deg: float = 285.0
    exhaust_open_deg: float = 795.0
    exhaust_close_deg: float = 1080.0

    @property
    def shape_factor(self) -> float:
        """Epitrochoid shape factor K = R / e [-]."""
        return self.rotor_radius / self.eccentricity

    @property
    def swept_volume(self) -> float:
        """Single-chamber swept displacement V_d = 3*sqrt(3)*R*e*b [m^3]."""
        return 3.0 * math.sqrt(3.0) * self.rotor_radius * self.eccentricity * self.rotor_width

    @property
    def min_volume(self) -> float:
        """Minimum clearance volume V_min = V_d / (CR - 1) [m^3]."""
        if self.compression_ratio <= 1.0:
            raise ValueError("Compression ratio must be strictly greater than 1.0.")
        return self.swept_volume / (self.compression_ratio - 1.0)

    @property
    def max_volume(self) -> float:
        """Maximum chamber volume V_max = V_min + V_d [m^3]."""
        return self.min_volume + self.swept_volume

    @property
    def theoretical_max_cr(self) -> float:
        """Theoretical epitrochoid compression limit without rotor pocket (Yamamoto 1981; Meng et al. 2023)."""
        k = max(self.shape_factor, 3.01)
        v_min_coeff = (
            math.pi / 3.0
            + 2.0 * math.sqrt(k * k - 9.0)
            - 1.5 * math.sqrt(3.0) * k
            + ((2.0 / 9.0) * k * k + 4.0) * math.asin(3.0 / k)
        )
        v_d_coeff = 3.0 * math.sqrt(3.0) * k
        return 1.0 + v_d_coeff / max(v_min_coeff, 1e-6)

    @property
    def pocket_volume(self) -> float:
        """Estimated rotor recess pocket volume V_k [m^3] above base geometric clearance."""
        # Geometric clearance without recess is small (~3-4% of V_d); remainder is rotor pocket V_k.
        v_geom_clearance = 0.038 * self.swept_volume
        return max(0.0, self.min_volume - v_geom_clearance)


@dataclass
class FuelBlendConfig:
    """Gasoline + hydrogen fuel blend and mixture settings.

    Attributes:
        h2_mass_fraction: Hydrogen mass fraction x_H2 = m_H2 / (m_gas + m_H2) in [0.0, 0.30].
        excess_air_ratio: Air-fuel equivalence / excess air ratio lambda >= 1.0.
        lhv_gasoline: Lower heating value of gasoline [J/kg] (44.0 MJ/kg).
        lhv_h2: Lower heating value of hydrogen [J/kg] (120.0 MJ/kg).
        afr_stoich_gasoline: Stoichiometric AFR for gasoline [-].
        afr_stoich_h2: Stoichiometric AFR for hydrogen [-].
    """

    h2_mass_fraction: float = 0.10
    excess_air_ratio: float = 1.15
    lhv_gasoline: float = 44.0e6
    lhv_h2: float = 120.0e6
    afr_stoich_gasoline: float = 14.7
    afr_stoich_h2: float = 34.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.h2_mass_fraction <= 0.30:
            raise ValueError(f"h2_mass_fraction must be in [0.0, 0.30], got {self.h2_mass_fraction}")
        if self.excess_air_ratio < 0.8:
            raise ValueError(f"excess_air_ratio lambda must be >= 0.8, got {self.excess_air_ratio}")

    @property
    def composite_lhv(self) -> float:
        """Mass-weighted Lower Heating Value LHV_mix [J/kg]."""
        return self.h2_mass_fraction * self.lhv_h2 + (1.0 - self.h2_mass_fraction) * self.lhv_gasoline

    @property
    def afr_stoich(self) -> float:
        """Mass-weighted stoichiometric air-fuel ratio [-]."""
        return (
            self.h2_mass_fraction * self.afr_stoich_h2
            + (1.0 - self.h2_mass_fraction) * self.afr_stoich_gasoline
        )

    @property
    def afr_actual(self) -> float:
        """Actual air-fuel ratio = lambda * AFR_stoich [-]."""
        return self.excess_air_ratio * self.afr_stoich

    @property
    def carbon_mass_fraction(self) -> float:
        """Fuel carbon mass fraction (iso-octane surrogate C8H18: 96/114 = 0.8421)."""
        return (1.0 - self.h2_mass_fraction) * (96.0 / 114.23)

    @property
    def h2_energy_fraction(self) -> float:
        """Fraction of total fuel chemical energy supplied by hydrogen [-]."""
        return (self.h2_mass_fraction * self.lhv_h2) / self.composite_lhv


@dataclass
class OperatingConditions:
    """Boundary and operating conditions for a single cycle simulation.

    Note on angle convention:
    In the 4-stroke Wankel eccentric shaft cycle (0-1080 deg):
    - Intake:      0 - 270 deg
    - Compression: 270 - 540 deg (TDC compression at 540 deg EA)
    - Expansion:   540 - 810 deg
    - Exhaust:     810 - 1080 deg
    Default spark timing is 522 deg EA (18 deg EA BTDC), consistent with
    Meng et al. (2023) reporting MBT CA50 ~ 35-40 deg EA ATDC.
    """

    rpm: float = 5000.0
    p_intake: float = 101325.0  # Pa (1 atm WOT)
    t_intake: float = 305.0  # K
    p_exhaust: float = 105000.0  # Pa (mild exhaust backpressure)
    spark_timing_deg: float = 522.0  # deg EA (18 deg BTDC relative to 540 deg TDC)
    residual_fraction_init: float = 0.06


@dataclass
class HybridGeneratorConfig:
    """Range-extender electrical load demand and generator efficiency model.

    Grounded in AVL / AIE 225CS range-extender specs (`Heat Transfer Project Information.pdf` §1.2):
    - Continuous target electrical power: 15.0 kW (expandable to 18.0 kW).
    - PMSM generator peak efficiency: ~91-92%.
    - Speed sweep range: 2,000 RPM to 7,000 RPM in 250-RPM increments.
    """

    target_electric_power_kw: float = 15.0
    peak_generator_efficiency: float = 0.92
    rpm_min: int = 2000
    rpm_max: int = 7000
    rpm_step: int = 250
    power_tolerance_kw: float = 0.15

    def generator_efficiency(self, rpm: float, p_brake_kw: float) -> float:
        """Evaluate PMSM generator conversion efficiency eta_gen(N, P_brake) [-].

        Models realistic high-efficiency permanent-magnet synchronous generator behavior:
        near 92% around nominal 4500-6000 RPM and 10-20 kW, dropping slightly at low
        speed (copper/switching losses) and very high speed (iron/windage losses).
        """
        if p_brake_kw <= 0.1:
            return 0.70
        # Speed penalty centered near 5250 RPM
        speed_norm = (rpm - 5250.0) / 3500.0
        speed_penalty = 0.025 * (speed_norm ** 2)
        # Load penalty below ~40% rated power (~7 kW)
        load_ratio = min(p_brake_kw / 16.5, 1.3)
        load_penalty = 0.04 * max(0.0, 0.55 - load_ratio) ** 2
        eta = self.peak_generator_efficiency - speed_penalty - load_penalty
        return max(0.80, min(0.94, eta))


@dataclass
class SimulationConfig:
    """Top-level configuration container for the simulation suite."""

    geometry: RotaryGeometry = field(default_factory=RotaryGeometry)
    fuel: FuelBlendConfig = field(default_factory=FuelBlendConfig)
    hybrid: HybridGeneratorConfig = field(default_factory=HybridGeneratorConfig)
    max_step_deg: float = 0.5
    ode_method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-8
