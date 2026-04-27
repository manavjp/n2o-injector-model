# ============================================================
# INJECTOR GEOMETRY & WATER-FLOW-DERIVED VARIABLES
# ============================================================
INJECTOR = {
    "name": "HF2 INJECTOR",                     # injector name
    "Cd": 0.6411441306,                         # CdA from water flow
    "A_physical": 0.00007348309828,             # m^2
    "L_over_D": 5.333310936,
}


# ============================================================
# OPERATING CONDITIONS (UPSTREAM)
# ============================================================
# Saturated upstream assumption: T1 is computed from saturation at P1 by CoolProp. 
OPERATING = {
    "P1_psi": 384.5364412,          # PT5 manifold avg
    "Pc_design_psi": 275,           # target chamber pressure
}


# ============================================================
# NHNE WEIGHTING (KAPPA)
# ============================================================
# kappa controls the NHNE blend between SPI and HEM.
#   - Large kappa  → SPI-dominant (short orifice, fast residence)
#   - Small kappa  → HEM-dominant (long orifice, full equilibrium)
# Sweep across multiple values; calibration finds the best fit.
KAPPA_SWEEP = [0.0]

# Single kappa for "primary" prediction (used in headline numbers, plot focus).
# Set to whichever kappa is currently treated as best-fit.
KAPPA_PRIMARY = 0.0


# ============================================================
# P2 SWEEP (DOWNSTREAM PRESSURE RANGE)
# ============================================================
SWEEP = {
    "P2_min_pa": 100000,                  # ambient floor (~14.5 psi)
    "N_points": 200,                      # mdot vs dP curve resolution
}


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================
# Used by calibrate.py only. Defines the sweep range and validation
# reference for finding the best-fit kappa.
CALIBRATION = {
    "kappa_min": 0.00001,                   # near-zero to capture HEM limit
    "kappa_max": 0.18,
    "kappa_steps": 50,                    # finer sweep
    "mdot_measured_lb_s": 1.85,           # m dot from load cell
    "burn_duration_s": 14,
    "total_ox_consumed_lb": 24.2,
    "pass_band_pct": 15,                  # percent margin
}


# ============================================================
# OUTPUT FILE NAMES
# ============================================================
# Files are written to ../outputs/ (or ../calibrations/) relative to script.
OUTPUT_FILES = {
    "csv":      "mdot_vs_dP.csv",
    "plot":     "mdot_vs_dP.png",
    "metadata": "run_metadata.json",
    "calibration_json":  "calibrated_params.json",
    "calibration_plot":  "calibration_kappa_sweep.png",
    "calibration_curves": "calibration_curves_at_best_kappa.png",
}