#!/usr/bin/env python3
"""
generate_racecar_ini.py
-----------------------
Generates a racecar.ini file compatible with the TUMFTM/global_racetrajectory_optimization
repository, tailored for an Electrathon vehicle.

The user is prompted for basic vehicle specs; all remaining parameters are filled
with sensible defaults for a small, low-power electric vehicle.

Usage:
    python generate_racecar_ini.py
"""

from typing import Optional


def prompt_float(label: str, unit: str, default: Optional[float] = None) -> float:
    """Prompt the user for a floating-point value with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label} ({unit}){suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("    ⚠  Please enter a valid number.")


def gather_inputs() -> dict:
    """Interactively collect Electrathon vehicle specs from the user."""
    print("\n" + "=" * 70)
    print("  TUMFTM racecar.ini Generator — Electrathon Edition")
    print("=" * 70)
    print("\nEnter your vehicle specifications below.\n")

    specs = {
        "mass":       prompt_float("Vehicle Mass",                    "kg"),
        "wheelbase":  prompt_float("Wheelbase",                       "m"),
        "lf":         prompt_float("Distance from Front Axle to CoG", "m"),
        "cog_z":      prompt_float("CoG Height",                      "m"),
        "dragcoeff":  prompt_float("Aerodynamic Drag Coefficient Cd", "-"),
        "frontal_a":  prompt_float("Frontal Area",                    "m²"),
        "power_max":  prompt_float("Max Motor Power",                 "W"),
    }
    return specs


def build_ini(specs: dict) -> str:
    """
    Return the complete racecar.ini text.

    Derived quantities
    ------------------
    * lr  = wheelbase − lf   (rear axle to CoG)
    * v_max estimated from P = 0.5 · Cd · A · ρ · v³  (aero-limited top speed)
    * f_drive_max = power_max / v_max  (peak traction force estimate)
    * f_brake_max ≈ mass · g  (1 g braking, conservative)
    * I_z approximated with  mass · wheelbase² / 12
    """
    mass      = specs["mass"]
    wheelbase = specs["wheelbase"]
    lf        = specs["lf"]
    lr        = round(wheelbase - lf, 4)
    cog_z     = specs["cog_z"]
    dragcoeff = specs["dragcoeff"]
    frontal_a = specs["frontal_a"]
    power_max = specs["power_max"]

    # ---------- derived ----------
    g = 9.81
    rho_air = 1.225  # kg/m³

    # Aero-limited top speed  P = 0.5 · Cd · A · ρ · v³
    v_max = round((2.0 * power_max / (dragcoeff * frontal_a * rho_air)) ** (1.0 / 3.0), 1)
    # Cap at a reasonable value for Electrathon (~15-30 m/s typical)
    v_max = min(v_max, 40.0)

    # Peak traction force at ~25 % of top speed (motor torque curve knee)
    f_drive_max = round(power_max / (v_max * 0.25), 1)
    f_brake_max = round(mass * g, 1)

    # Yaw inertia approximation
    I_z = round(mass * wheelbase ** 2 / 12.0, 1)

    # Nominal tyre load per axle (static 50/50 assumed)
    f_z0 = round(mass * g / 4.0, 1)  # per-corner

    # Track width — Electrathon typical ~1.0 m
    track_w = 1.0

    # Vehicle "length" for the optimizer (approx wheelbase + bumpers)
    length_veh = round(wheelbase + 0.3, 2)

    # Width (including small safety margin)
    width_veh = round(track_w + 0.2, 2)
    width_opt = round(width_veh + 0.4, 2)

    # Curvature limit
    curvlim = 0.12

    # Effective drag coefficient for TUMFTM format  (0.5 * Cd * A * rho) / mass
    # The repo uses  dragcoeff  as:  F_drag = dragcoeff * v²   → dragcoeff = 0.5 * Cd * A * rho
    tumftm_dragcoeff = round(0.5 * dragcoeff * frontal_a * rho_air, 6)

    # Wheel radius — small kart-style wheels
    r_wheel = 0.25

    # Max steer angle
    delta_max = 0.50  # rad (~28 deg)

    ini = f"""\
# ----------------------------------------------------------------------------------------------------------------------
# TUMFTM racecar.ini — Auto-generated for Electrathon vehicle
# Generator: generate_racecar_ini.py
# ----------------------------------------------------------------------------------------------------------------------

[GENERAL_OPTIONS]

# --- file references for ggv diagram and ax_max_machines ---
ggv_file="ggv.csv"
ax_max_machines_file="ax_max_machines.csv"

# stepsize_prep:                [m] used for track preprocessing (sampling)
# stepsize_reg:                 [m] used for spline regularisation
# stepsize_interp_after_opt:    [m] used for spline interpolation after optimization
stepsize_opts={{"stepsize_prep": 1.0,
               "stepsize_reg": 3.0,
               "stepsize_interp_after_opt": 2.0}}

# k_reg:                        [-] order of B-spline
# s_reg:                        [-] smoothing factor, range [1.0, 100.0]
reg_smooth_opts={{"k_reg": 3,
                 "s_reg": 10}}

# d_preview_curv:               [m] preview distance (curvature)
# d_review_curv:                [m] review distance (curvature)
# d_preview_head:               [m] preview distance (heading)
# d_review_head:                [m] review distance (heading)
curv_calc_opts = {{"d_preview_curv": 2.0,
                  "d_review_curv": 2.0,
                  "d_preview_head": 1.0,
                  "d_review_head": 1.0}}

# v_max:                        [m/s] maximum vehicle speed
# length:                       [m] vehicle length
# width:                        [m] vehicle width
# mass:                         [kg] vehicle mass
# dragcoeff:                    [kg/m] drag coefficient  (0.5 * Cd * A * rho)
# curvlim:                      [rad/m] curvature limit of the battery-electric vehicle
# g:                            [N/kg] gravity acceleration
veh_params = {{"v_max": {v_max},
              "length": {length_veh},
              "width": {width_veh},
              "mass": {mass},
              "dragcoeff": {tumftm_dragcoeff},
              "curvlim": {curvlim},
              "g": {g}}}

# dyn_model_exp:                [-] exponent used in the dynamic model (1.0 = linear)
# vel_profile_conv_filt_window: [-] moving average filter window size (set null if not used)
vel_calc_opts = {{"dyn_model_exp": 1.0,
                 "vel_profile_conv_filt_window": null}}

# ----------------------------------------------------------------------------------------------------------------------
[OPTIMIZATION_OPTIONS]

# width_opt:                    [m] vehicle width for optimization including safety distance
optim_opts_shortest_path={{"width_opt": {width_opt}}}

# iqp_curverror_allowed:        [rad/m] maximum allowed curvature error for the IQP
optim_opts_mincurv={{"width_opt": {width_opt},
                    "iqp_iters_min": 3,
                    "iqp_curverror_allowed": 0.01}}

# Minimum-time optimisation parameters
optim_opts_mintime={{"width_opt": {width_opt},
                    "penalty_delta": 10.0,
                    "penalty_F": 0.01,
                    "mue": 1.0,
                    "n_gauss": 5,
                    "dn": 0.25,
                    "limit_energy": false,
                    "energy_limit": 2.0,
                    "safe_traj": false,
                    "ax_pos_safe": null,
                    "ax_neg_safe": null,
                    "ay_safe": null,
                    "w_tr_reopt": 2.0,
                    "w_veh_reopt": 1.6,
                    "w_add_spl_regr": 0.2,
                    "step_non_reg": 0,
                    "eps_kappa": 1e-3}}

# Vehicle parameters for minimum-time optimisation
# wheelbase_front:              [m] front axle to CoG
# wheelbase_rear:               [m] rear axle to CoG
# track_width_front/rear:       [m] track width
# cog_z:                        [m] centre of gravity height
# I_z:                          [kg·m²] yaw moment of inertia
# liftcoeff_front/rear:         [-] aerodynamic lift (downforce) coefficients
# k_brake_front:                [-] fraction of braking force on front axle
# k_drive_front:                [-] fraction of drive force on front axle (0 = rear-wheel drive)
# k_roll:                       [-] roll stiffness distribution (0.5 = even)
# t_delta:                      [s] steering actuator time constant
# t_drive:                      [s] drive actuator time constant
# t_brake:                      [s] brake actuator time constant
# power_max:                    [W] maximum motor power
# f_drive_max:                  [N] maximum drive force
# f_brake_max:                  [N] maximum brake force
# delta_max:                    [rad] maximal steer angle
vehicle_params_mintime = {{"wheelbase_front": {lf},
                          "wheelbase_rear": {lr},
                          "track_width_front": {track_w},
                          "track_width_rear": {track_w},
                          "cog_z": {cog_z},
                          "I_z": {I_z},
                          "liftcoeff_front": 0.0,
                          "liftcoeff_rear": 0.0,
                          "k_brake_front": 0.6,
                          "k_drive_front": 0.0,
                          "k_roll": 0.5,
                          "t_delta": 0.2,
                          "t_drive": 0.05,
                          "t_brake": 0.05,
                          "power_max": {power_max},
                          "f_drive_max": {f_drive_max},
                          "f_brake_max": {f_brake_max},
                          "delta_max": {delta_max}}}

# Tyre parameters (Pacejka Magic Formula defaults for small bias-ply kart tyres)
# c_roll:                       [-] rolling resistance coefficient
# f_z0:                         [N] nominal tyre load (per corner)
# B / C / eps / E:              [-] Magic Formula coefficients (front & rear)
tire_params_mintime = {{"c_roll": 0.015,
                       "f_z0": {f_z0},
                       "B_front": 10.0,
                       "C_front": 2.5,
                       "eps_front": -0.1,
                       "E_front": 1.0,
                       "B_rear": 10.0,
                       "C_rear": 2.5,
                       "eps_rear": -0.1,
                       "E_rear": 1.0}}

# Power / thermal parameters for minimum-time simulation
# pwr_behavior=false disables the detailed thermal model
pwr_params_mintime = {{"pwr_behavior": false,
                      "simple_loss": true,
                      "T_env": 30,
                      "T_mot_ini": 30,
                      "T_batt_ini": 30,
                      "T_inv_ini": 30,
                      "T_cool_mi_ini": 30,
                      "T_cool_b_ini": 30,
                      "r_wheel": {r_wheel},
                      "R_i_sumo": 0.001,
                      "R_i_simple": 0.5,
                      "R_i_offset": 0.005,
                      "R_i_slope": 1e-5,
                      "V_OC_simple": 48.0,
                      "SOC_ini": 0.9,
                      "C_batt": 20.0,
                      "N_cells_serial": 13,
                      "N_cells_parallel": 1,
                      "temp_mot_max": 120.0,
                      "temp_batt_max": 50.0,
                      "temp_inv_max": 80.0,
                      "N_machines": 1,
                      "transmission": 10.0,
                      "MotorConstant": 0.30,
                      "C_therm_machine": 3000.0,
                      "C_therm_inv": 2000.0,
                      "C_therm_cell": 500.0,
                      "C_TempCopper": 0.004041,
                      "m_therm_fluid_mi": 2,
                      "m_therm_fluid_b": 2,
                      "R_Phase": 0.05,
                      "r_rotor_int": 0.02,
                      "r_rotor_ext": 0.06,
                      "r_stator_int": 0.061,
                      "r_stator_ext": 0.09,
                      "l_machine": 0.04,
                      "A_cool_inflate_machine": 1.0,
                      "A_cool_inv": 0.1,
                      "A_cool_rad": 1.0,
                      "k_iro": 45.0,
                      "h_air": 50.0,
                      "h_air_gap": 60.0,
                      "h_fluid_mi": 5000.0,
                      "c_heat_fluid": 4181.0,
                      "flow_rate_inv": 0.1,
                      "flow_rate_rad": 0.1,
                      "machine_simple_a": -0.000027510784764,
                      "machine_simple_b": 1.046187222759047,
                      "machine_simple_c": 1.001964003837042,
                      "V_ref": 48.0,
                      "I_ref": 100.0,
                      "V_ce_offset": 0.8,
                      "V_ce_slope": 0.0036,
                      "E_on": 0.022,
                      "E_off": 0.057,
                      "E_rr": 0.04,
                      "f_sw": 12000.0,
                      "inverter_simple_a": -0.000707138661579,
                      "inverter_simple_b": 1.139958410466637,
                      "inverter_simple_c": 1.004970807882952}}
"""
    return ini


def main():
    specs = gather_inputs()
    ini_text = build_ini(specs)

    out_path = "racecar.ini"
    with open(out_path, "w") as f:
        f.write(ini_text)

    print(f"\n✅  racecar.ini written to  →  {out_path}")
    print("   Copy this file into the  params/  folder of the TUMFTM repository.\n")


if __name__ == "__main__":
    main()
