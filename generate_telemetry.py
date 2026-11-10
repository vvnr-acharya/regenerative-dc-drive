#!/usr/bin/env python3
"""
===============================================================================
PROJECT: Real-Time Telemetry, Edge Analytics, and 3D Digital Twin for a
         Regenerative DC Drive Powertrain
MODULE 1: DATA ANALYTICS & FIRST-PRINCIPLES SIMULATION ENGINE
SCRIPT:   generate_telemetry.py
===============================================================================

Domain: Electrical & Electronics Engineering (EEE) Powertrain Data Analytics
Hardware Target:
  - Custom H-Bridge Motor Driver:
      * High-Side: IRF9540N P-channel MOSFETs (V_DSS = -100V, R_DS(on) = 0.117 Ohm, V_SD = 1.15V)
      * Low-Side:  IRFZ44N N-channel MOSFETs (V_DSS = 55V, R_DS(on) = 0.0175 Ohm)
      * Gate Drive: BC547 BJT Level-Shifters
      * Capacitor Bank: 1000 uF / 25V Low-ESR Bus Capacitor (C1)
      * Snubber: 100 nF High-Frequency Ceramic Snubber
      * Current Sensor: 0.05 Ohm Precision Copper Wire Shunt
  - DC Bus: 12V Nominal Rail (11.4V to 12.6V working window)
  - Switching: 15 kHz Dynamic Braking Low-Side PWM Boost Regeneration

This module implements a rigorous, first-principles electromechanical simulation
engine modeling the four physical phases of DC drive operation:
  Phase 1: Motoring (Steady-state entry speed 800 to 3,500 RPM)
  Phase 2: Coasting (Gate driver idle, back-EMF terminal sensing)
  Phase 3: Boost Braking (15 kHz low-side PWM inductive boost energy transfer)
  Phase 4: Halt (Deceleration until speed <= 10 RPM)

Calculated Features:
  - initial_rpm: Pre-braking steady entry shaft speed [RPM]
  - pwm_brake_duty: Low-side switching duty cycle [0.10 to 0.90]
  - bus_voltage_init: Pre-braking DC bus supply voltage [V]
  - motor_temp_c: Armature winding core temperature [deg C]
  - peak_current_a: Maximum peak armature current observed during braking [A]
  - peak_kickback_voltage_v: Peak inductive spike voltage on terminal/bus [V]
  - mean_current_ripple: Average high-frequency current ripple across cycles [A]

Target Variables:
  1. energy_recovered_joules (Continuous: Integral of V_bus(t)*I_regen(t) dt - I_a^2(t)*R_a dt)
  2. stopping_time_ms (Continuous: Duration from brake onset until RPM <= 10)
  3. safety_class (Categorical: 0 = Inefficient, 1 = Optimal Safe Regen, 2 = Overvoltage/Thermal Hazard)
"""

import os
import sys
import time
import ctypes
import tempfile
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1. ELECTROMECHANICAL & CIRCUIT PARAMETERS (FIRST-PRINCIPLES SPECIFICATION)
# =============================================================================

@dataclass(frozen=True)
class MotorParameters:
    """Brushed DC Motor electromechanical constants."""
    Ra_25: float = 1.8           # Armature resistance at 25 deg C [Ohm]
    alpha_Cu: float = 0.00393    # Copper temperature coefficient [1/deg C]
    La: float = 2.5e-3           # Armature inductance [H] (2.5 mH)
    Kt: float = 0.045            # Torque constant [N*m/A]
    Ke: float = 0.045            # Back-EMF constant [V/(rad/s)]
    J: float = 0.0008            # Rotor & flywheel moment of inertia [kg*m^2]
    B: float = 0.0001            # Viscous damping coefficient [N*m*s/rad]
    tau_load: float = 0.045      # Test-bench parasitic/load friction [N*m]


@dataclass(frozen=True)
class HBridgeParameters:
    """Custom discrete MOSFET H-Bridge driver parameters."""
    Rds_on_low: float = 0.0175   # IRFZ44N Low-side N-channel MOSFET on-resistance [Ohm]
    Rds_on_high: float = 0.117   # IRF9540N High-side P-channel MOSFET on-resistance [Ohm]
    V_diode: float = 1.15        # High-side P-channel body diode forward drop [V]
    R_shunt: float = 0.05        # Current sensing shunt resistance [Ohm]
    C_bus: float = 1000e-6       # Bus capacitor bank C1 capacitance [F] (1000 uF)
    C_snubber: float = 100e-9    # Ceramic snubber capacitance [F] (100 nF)
    R_bat_internal: float = 0.08 # Battery lead-acid/LiFePO4 internal resistance [Ohm]
    f_pwm: float = 15000.0       # Braking PWM frequency [Hz] (15 kHz)


# =============================================================================
# 2. C-ACCELERATION ENGINE (HIGH-PERFORMANCE MULTI-RUN SIMULATOR)
# =============================================================================

C_SIMULATION_SOURCE = r"""
#include <math.h>
#include <stdlib.h>

void simulate_batch(
    int n,
    const double* rpms,
    const double* duties,
    const double* vbuses,
    const double* temps,
    double* out_stopping_time_ms,
    double* out_peak_current_a,
    double* out_peak_kickback_v,
    double* out_mean_ripple,
    double* out_energy_recovered_j,
    int* out_safety_class
) {
    // Electromechanical constants
    const double La = 2.5e-3;
    const double Kt = 0.045;
    const double Ke = 0.045;
    const double J = 0.0008;
    const double B = 0.0001;
    const double tau_load = 0.045;

    // Circuit hardware constants
    const double R_bat = 0.08;
    const double Vd = 1.15;
    const double f_pwm = 15000.0;
    const double T = 1.0 / f_pwm;
    const double omega_halt = 10.0 * 2.0 * M_PI / 60.0; // 10 RPM threshold

    for (int i = 0; i < n; i++) {
        double rpm = rpms[i];
        double duty = duties[i];
        double vbus_init = vbuses[i];
        double temp = temps[i];

        // Temperature-dependent copper winding resistance
        double Ra = 1.8 * (1.0 + 0.00393 * (temp - 25.0));

        // Loop resistances
        // ON Phase: Motor -> Low-side FET -> Shunt -> GND -> Low-side FET -> Motor
        double Ron = Ra + (0.0175 * 2.0) + 0.05;
        // OFF Phase: Motor -> High-side body diode -> Bus -> Ground -> Low-side return
        double Roff = Ra + 0.0175 + 0.05;

        double t_on = duty * T;
        double t_off = (1.0 - duty) * T;
        double tau_on = La / Ron;
        double tau_off = La / Roff;

        double exp_on = exp(-t_on / tau_on);
        double exp_off = exp(-t_off / tau_off);

        double omega = rpm * 2.0 * M_PI / 60.0;
        double ia = 0.0;
        double Vbus = vbus_init;
        double t = 0.0;
        double imax = 0.0;
        double vkick_max = vbus_init;
        double erec = 0.0;
        double ripple_sum = 0.0;
        long ripple_count = 0;

        const double max_t = 3.8; // Safety ceiling [s]

        while (omega > omega_halt && t < max_t) {
            double eb = Ke * omega;

            // 1. ON PHASE: Low-side MOSFET pulsed ON (Inductor magnetic charging)
            double Iss_on = eb / Ron;
            double i_on = Iss_on + (ia - Iss_on) * exp_on;
            double i_on_avg = Iss_on + (ia - Iss_on) * (tau_on / t_on) * (1.0 - exp_on);

            // 2. OFF PHASE: Low-side MOSFET OFF (Flyback inductive boost into bus)
            double Iss_off = (eb - (Vbus + Vd)) / Roff;
            double q_off = 0.0;
            double i_next = 0.0;

            if (Iss_off < 0.0) {
                // Potential Discontinuous Conduction Mode (DCM)
                double denom = i_on - Iss_off + 1e-12;
                double val = -Iss_off / denom;
                if (val > 0.0 && val < 1.0) {
                    double t_cond = -tau_off * log(val);
                    if (t_cond < t_off) {
                        double exp_c = exp(-t_cond / tau_off);
                        q_off = Iss_off * t_cond + (i_on - Iss_off) * tau_off * (1.0 - exp_c);
                        i_next = 0.0;
                    } else {
                        q_off = Iss_off * t_off + (i_on - Iss_off) * tau_off * (1.0 - exp_off);
                        i_next = Iss_off + (i_on - Iss_off) * exp_off;
                    }
                } else {
                    q_off = Iss_off * t_off + (i_on - Iss_off) * tau_off * (1.0 - exp_off);
                    i_next = Iss_off + (i_on - Iss_off) * exp_off;
                }
            } else {
                // Continuous Conduction Mode (CCM)
                q_off = Iss_off * t_off + (i_on - Iss_off) * tau_off * (1.0 - exp_off);
                i_next = Iss_off + (i_on - Iss_off) * exp_off;
            }

            // Current metrics tracking
            double cycle_max = (ia > i_on) ? ia : i_on;
            if (cycle_max > imax) imax = cycle_max;

            double cycle_min = (ia < i_next) ? ia : i_next;
            ripple_sum += (cycle_max - cycle_min);
            ripple_count++;

            // Regenerative charge injection into bus capacitor bank C1 & battery rail
            double I_regen = (q_off > 0.0) ? (q_off / T) : 0.0;
            Vbus = vbus_init + R_bat * I_regen;

            // Inductive kickback voltage at motor terminal during switch turn-off
            // Clamped by high-side body diode with snubber attenuation
            double v_kick = Vbus + Vd + (0.16 * cycle_max);
            if (v_kick > vkick_max) vkick_max = v_kick;

            // Mechanical dynamics integration
            double i_avg = (i_on_avg * t_on + q_off) / T;
            if (i_avg < 0.0) i_avg = 0.0;

            double torque_elec = Kt * i_avg;
            double torque_total = torque_elec + B * omega + tau_load;

            omega -= (torque_total / J) * T;
            if (omega < 0.0) omega = 0.0;

            // Energy accounting: Net recovered energy = Bus energy minus copper heating
            erec += (Vbus * I_regen - (i_avg * i_avg) * Ra) * T;
            ia = (i_next > 0.0) ? i_next : 0.0;
            t += T;
        }

        double t_stop_ms = t * 1000.0;
        double mean_ripple = (ripple_count > 0) ? (ripple_sum / ripple_count) : 0.0;

        // Categorical Safety Classification
        // 0 = Inefficient (insufficient regen, sluggish stopping, weak duty)
        // 1 = Optimal Safe Regen (positive recovery, safe voltage/current, controlled halt)
        // 2 = Overvoltage / Thermal Hazard (hazardous kickback spike, excessive current surge)
        int s_class = 1;
        if (vkick_max >= 14.05 || imax >= 4.8 || (temp >= 74.0 && imax >= 2.8)) {
            s_class = 2; // Overvoltage / Thermal Hazard
        } else if (erec <= 0.20 || t_stop_ms >= 2700.0 || duty <= 0.25) {
            s_class = 0; // Inefficient
        } else {
            s_class = 1; // Optimal Safe Regen
        }

        out_stopping_time_ms[i] = t_stop_ms;
        out_peak_current_a[i] = imax;
        out_peak_kickback_v[i] = vkick_max;
        out_mean_ripple[i] = mean_ripple;
        out_energy_recovered_j[i] = erec;
        out_safety_class[i] = s_class;
    }
}
"""


class FastPowertrainSimulator:
    """
    High-performance simulation driver utilizing automated JIT C compilation
    with pure Python/NumPy fallback.
    """

    def __init__(self):
        self._c_lib = None
        self._compile_c_engine()

    def _compile_c_engine(self) -> None:
        """Compiles the standalone C engine into a shared library."""
        try:
            tmp_dir = tempfile.gettempdir()
            c_path = os.path.join(tmp_dir, "sim_engine_powertrain.c")
            so_path = os.path.join(tmp_dir, "libsim_engine_powertrain.so")

            with open(c_path, "w", encoding="utf-8") as f:
                f.write(C_SIMULATION_SOURCE)

            cmd = ["gcc", "-O3", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

            if res.returncode == 0 and os.path.exists(so_path):
                c_lib = ctypes.CDLL(so_path)
                c_lib.simulate_batch.argtypes = [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
                ]
                self._c_lib = c_lib
        except Exception:
            self._c_lib = None

    def run_simulation(
        self,
        rpms: np.ndarray,
        duties: np.ndarray,
        vbuses: np.ndarray,
        temps: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Executes first-principles simulation across N distinct parameter sets.
        """
        n = len(rpms)
        if self._c_lib is not None:
            out_stop = np.zeros(n, dtype=np.float64)
            out_imax = np.zeros(n, dtype=np.float64)
            out_vkick = np.zeros(n, dtype=np.float64)
            out_ripple = np.zeros(n, dtype=np.float64)
            out_erec = np.zeros(n, dtype=np.float64)
            out_class = np.zeros(n, dtype=np.int32)

            self._c_lib.simulate_batch(
                ctypes.c_int(n),
                rpms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                duties.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                vbuses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                temps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_stop.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_imax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_vkick.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_ripple.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_erec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_class.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )

            return {
                "stopping_time_ms": out_stop,
                "peak_current_a": out_imax,
                "peak_kickback_voltage_v": out_vkick,
                "mean_current_ripple": out_ripple,
                "energy_recovered_joules": out_erec,
                "safety_class": out_class
            }
        else:
            return self._run_python_fallback(rpms, duties, vbuses, temps)

    def _run_python_fallback(
        self,
        rpms: np.ndarray,
        duties: np.ndarray,
        vbuses: np.ndarray,
        temps: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Pure-Python vectorized fallback simulation."""
        n = len(rpms)
        out_stop = np.zeros(n)
        out_imax = np.zeros(n)
        out_vkick = np.zeros(n)
        out_ripple = np.zeros(n)
        out_erec = np.zeros(n)
        out_class = np.zeros(n, dtype=int)

        La = 2.5e-3
        Kt = 0.045
        Ke = 0.045
        J = 0.0008
        B = 0.0001
        tau_load = 0.045
        R_bat = 0.08
        Vd = 1.15
        f_pwm = 15000.0
        T = 1.0 / f_pwm
        omega_halt = 10.0 * 2.0 * np.pi / 60.0

        for i in range(n):
            rpm = rpms[i]
            duty = duties[i]
            vbus_init = vbuses[i]
            temp = temps[i]

            Ra = 1.8 * (1.0 + 0.00393 * (temp - 25.0))
            Ron = Ra + (0.0175 * 2.0) + 0.05
            Roff = Ra + 0.0175 + 0.05

            t_on = duty * T
            t_off = (1.0 - duty) * T
            tau_on = La / Ron
            tau_off = La / Roff
            exp_on = np.exp(-t_on / tau_on)
            exp_off = np.exp(-t_off / tau_off)

            omega = rpm * 2.0 * np.pi / 60.0
            ia = 0.0
            Vbus = vbus_init
            t = 0.0
            imax = 0.0
            vkick_max = vbus_init
            erec = 0.0
            ripples = []

            while omega > omega_halt and t < 3.8:
                eb = Ke * omega
                Iss_on = eb / Ron
                i_on = Iss_on + (ia - Iss_on) * exp_on
                i_on_avg = Iss_on + (ia - Iss_on) * (tau_on / t_on) * (1.0 - exp_on)

                Iss_off = (eb - (Vbus + Vd)) / Roff
                val = -Iss_off / (i_on - Iss_off + 1e-12)

                if Iss_off < 0 and 0.0 < val < 1.0:
                    t_cond = min(t_off, -tau_off * np.log(val))
                    exp_c = np.exp(-t_cond / tau_off)
                    q_off = Iss_off * t_cond + (i_on - Iss_off) * tau_off * (1.0 - exp_c)
                    i_next = 0.0 if t_cond < t_off else Iss_off + (i_on - Iss_off) * exp_c
                else:
                    q_off = Iss_off * t_off + (i_on - Iss_off) * tau_off * (1.0 - exp_off)
                    i_next = Iss_off + (i_on - Iss_off) * exp_off

                cycle_max = max(ia, i_on)
                if cycle_max > imax:
                    imax = cycle_max

                cycle_min = min(ia, i_next)
                ripples.append(cycle_max - cycle_min)

                I_regen = max(0.0, q_off / T)
                Vbus = vbus_init + R_bat * I_regen
                v_kick = Vbus + Vd + (0.16 * cycle_max)
                if v_kick > vkick_max:
                    vkick_max = v_kick

                i_avg = (i_on_avg * t_on + q_off) / T
                torque_elec = Kt * max(0.0, i_avg)
                torque_total = torque_elec + B * omega + tau_load
                omega = max(0.0, omega - (torque_total / J) * T)

                erec += (Vbus * I_regen - (max(0.0, i_avg) ** 2) * Ra) * T
                ia = max(0.0, i_next)
                t += T

            t_stop_ms = t * 1000.0
            mean_rip = np.mean(ripples) if ripples else 0.0

            s_class = 1
            if vkick_max >= 14.05 or imax >= 4.8 or (temp >= 74.0 and imax >= 2.8):
                s_class = 2
            elif erec <= 0.20 or t_stop_ms >= 2700.0 or duty <= 0.25:
                s_class = 0

            out_stop[i] = t_stop_ms
            out_imax[i] = imax
            out_vkick[i] = vkick_max
            out_ripple[i] = mean_rip
            out_erec[i] = erec
            out_class[i] = s_class

        return {
            "stopping_time_ms": out_stop,
            "peak_current_a": out_imax,
            "peak_kickback_voltage_v": out_vkick,
            "mean_current_ripple": out_ripple,
            "energy_recovered_joules": out_erec,
            "safety_class": out_class
        }


# =============================================================================
# 3. SENSOR NOISE MODELING & DATASET GENERATION PIPELINE
# =============================================================================

def add_sensor_noise(
    peak_current: np.ndarray,
    peak_kickback: np.ndarray,
    mean_ripple: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates realistic embedded telemetry sensor noise:
      - 0.05 Ohm copper shunt: Thermal noise + instrumentation amp offset + ADC quantization
      - Voltage divider: Resistor tolerance noise + ADC LSB jitter
      - Current ripple: High-frequency 15 kHz PWM switching ripple harmonics
    """
    # Current shunt sensor noise: Gaussian sigma = 0.025 A (equivalent to ~1.25 mV on 0.05 Ohm shunt)
    shunt_noise = rng.normal(0.0, 0.025, size=peak_current.shape)
    noisy_peak_current = np.clip(peak_current + shunt_noise, 0.0, None)

    # Voltage divider sensor noise: Gaussian sigma = 0.035 V (resistive attenuation noise)
    vdiv_noise = rng.normal(0.0, 0.035, size=peak_kickback.shape)
    noisy_peak_kickback = np.clip(peak_kickback + vdiv_noise, 0.0, None)

    # Ripple measurement noise: Gaussian sigma = 0.002 A
    ripple_noise = rng.normal(0.0, 0.002, size=mean_ripple.shape)
    noisy_mean_ripple = np.clip(mean_ripple + ripple_noise, 0.0, None)

    return (
        np.round(noisy_peak_current, 4),
        np.round(noisy_peak_kickback, 4),
        np.round(noisy_mean_ripple, 5)
    )


def generate_dataset(
    num_samples: int = 1500,
    seed: int = 42,
    output_path: str = "motor_regen_dataset.csv"
) -> pd.DataFrame:
    """
    Generates the complete 1,500-run regenerative braking dataset,
    validates data integrity, and exports cleanly to CSV.
    """
    rng = np.random.default_rng(seed)

    print(f"[*] Initializing simulation of {num_samples} distinct electromechanical test runs...")
    start_time = time.time()

    # 1. Parameter sampling across specified academic & hardware operational boundaries
    initial_rpm = rng.uniform(800.0, 3500.0, size=num_samples)
    pwm_brake_duty = rng.uniform(0.10, 0.90, size=num_samples)
    bus_voltage_init = rng.uniform(11.4, 12.6, size=num_samples)
    motor_temp_c = rng.uniform(25.0, 85.0, size=num_samples)

    # 2. First-principles simulation execution
    sim = FastPowertrainSimulator()
    raw_results = sim.run_simulation(
        initial_rpm,
        pwm_brake_duty,
        bus_voltage_init,
        motor_temp_c
    )

    elapsed_time = time.time() - start_time
    print(f"[+] Simulation completed in {elapsed_time:.3f} seconds ({elapsed_time / num_samples * 1000:.2f} ms/run).")

    # 3. Add realistic sensor noise (Gaussian noise + 15 kHz PWM ripple)
    noisy_peak_i, noisy_peak_v, noisy_ripple = add_sensor_noise(
        raw_results["peak_current_a"],
        raw_results["peak_kickback_voltage_v"],
        raw_results["mean_current_ripple"],
        rng
    )

    # 4. Construct dataset DataFrame
    df = pd.DataFrame({
        "initial_rpm": np.round(initial_rpm, 2),
        "pwm_brake_duty": np.round(pwm_brake_duty, 4),
        "bus_voltage_init": np.round(bus_voltage_init, 3),
        "motor_temp_c": np.round(motor_temp_c, 2),
        "peak_current_a": noisy_peak_i,
        "peak_kickback_voltage_v": noisy_peak_v,
        "mean_current_ripple": noisy_ripple,
        "energy_recovered_joules": np.round(raw_results["energy_recovered_joules"], 4),
        "stopping_time_ms": np.round(raw_results["stopping_time_ms"], 2),
        "safety_class": raw_results["safety_class"]
    })

    # 5. Data sanitization & schema validation
    assert len(df) == num_samples, f"Sample count mismatch: expected {num_samples}, got {len(df)}"
    assert not df.isnull().any().any(), "Dataset contains null or NaN values!"
    assert not np.isinf(df.to_numpy()).any(), "Dataset contains infinite values!"

    expected_cols = [
        "initial_rpm",
        "pwm_brake_duty",
        "bus_voltage_init",
        "motor_temp_c",
        "peak_current_a",
        "peak_kickback_voltage_v",
        "mean_current_ripple",
        "energy_recovered_joules",
        "stopping_time_ms",
        "safety_class"
    ]
    assert list(df.columns) == expected_cols, "Column schema mismatch!"

    # 6. Save clean dataset to disk
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset successfully exported to: {output_path}")

    # Also save to artifacts folder if it exists
    alt_dir = "artifacts/file_generation/ttl=63d/output"
    if os.path.exists(alt_dir) or os.path.exists("artifacts"):
        os.makedirs(alt_dir, exist_ok=True)
        alt_path = os.path.join(alt_dir, "motor_regen_dataset.csv")
        df.to_csv(alt_path, index=False)
        print(f"[+] Artifact copy exported to: {alt_path}")

    # 7. Print statistical summary and safety class distribution
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION & DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(f"Total Records: {len(df):,}")
    print("\nCategorical Target Distribution ('safety_class'):")
    class_names = {
        0: "0 (Inefficient / Sluggish)",
        1: "1 (Optimal Safe Regen)",
        2: "2 (Overvoltage / Thermal Hazard)"
    }
    class_counts = df["safety_class"].value_counts().sort_index()
    for cls_idx, count in class_counts.items():
        pct = (count / len(df)) * 100.0
        print(f"  Class {class_names.get(cls_idx, str(cls_idx))}: {count:5d} runs ({pct:5.2f}%)")

    print("\nContinuous Feature & Target Ranges:")
    print(f"  initial_rpm             : [{df['initial_rpm'].min():7.1f}, {df['initial_rpm'].max():7.1f}] RPM")
    print(f"  pwm_brake_duty          : [{df['pwm_brake_duty'].min():7.4f}, {df['pwm_brake_duty'].max():7.4f}]")
    print(f"  bus_voltage_init        : [{df['bus_voltage_init'].min():7.3f}, {df['bus_voltage_init'].max():7.3f}] V")
    print(f"  motor_temp_c            : [{df['motor_temp_c'].min():7.2f}, {df['motor_temp_c'].max():7.2f}] deg C")
    print(f"  peak_current_a          : [{df['peak_current_a'].min():7.4f}, {df['peak_current_a'].max():7.4f}] A")
    print(f"  peak_kickback_voltage_v : [{df['peak_kickback_voltage_v'].min():7.4f}, {df['peak_kickback_voltage_v'].max():7.4f}] V")
    print(f"  mean_current_ripple     : [{df['mean_current_ripple'].min():7.5f}, {df['mean_current_ripple'].max():7.5f}] A")
    print(f"  energy_recovered_joules : [{df['energy_recovered_joules'].min():7.4f}, {df['energy_recovered_joules'].max():7.4f}] J")
    print(f"  stopping_time_ms        : [{df['stopping_time_ms'].min():7.2f}, {df['stopping_time_ms'].max():7.2f}] ms")
    print("=" * 70 + "\n")

    return df


# =============================================================================
# 4. COMMAND-LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="First-Principles DC Motor Regenerative Powertrain Telemetry Simulator"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1500,
        help="Number of distinct test runs to simulate (default: 1500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible parameter sampling (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="motor_regen_dataset.csv",
        help="Target output CSV file path (default: motor_regen_dataset.csv)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        num_samples=args.samples,
        seed=args.seed,
        output_path=args.output
    )