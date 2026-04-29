import math


# ============================================================
# INJECTORS
# ============================================================
INJECTORS = {
    "HF2_old": {
        "name": "HF2 INJECTOR",
        "D_m": 0.00119063,                    # hole diameter (m)
        "N_holes": 66,                        # number of orifices (count)
        "L_m": 0.00635,                       # hole depth / plate thickness (m)
        "Cd_water": 0.6411441306,             # discharge coefficient from water flow
    },

    "new": {
        "name": "HF2 UNFIRED INJ.", # this is with pilot holes
        "D_m": 0.001,                         # hole diameter (m)
        "N_holes": 36,                        # number of orifices (count)
        "L_m": 0.00635,                       # hole depth / plate thickness (m)
        "Cd_water": 0.6879905414,             # discharge coefficient from water flow
    },
}

SELECTED_INJECTOR = "new"       # CHOOSE THE INJECTOR


# ============================================================
# CALIBRATION
# ============================================================
EMPIRICAL_CORRECTIONS = {
    # calibrate.py overwrites this
    "Cd_correction_factor_HF2": 0.9043,    # = mdot_measured / mdot_predicted
}

# ============================================================
# DERIVED PROPERTIES (do not edit)
# ============================================================
def _build_injector(entry):
    """Compute A_physical, L/D, Cd from raw geometry inputs."""
    if any(entry.get(k) is None for k in ("D_m", "N_holes", "L_m", "Cd_water")):
        return {
            "name": entry.get("name", "UNFILLED"),
            "D_m": entry.get("D_m"),
            "N_holes": entry.get("N_holes"),
            "L_m": entry.get("L_m"),
            "Cd_water": entry.get("Cd_water"),
            "Cd": None,
            "A_physical": None,
            "L_over_D": None,
            "complete": False,
        }
    A_per_hole = math.pi / 4 * entry["D_m"] ** 2
    A_physical = entry["N_holes"] * A_per_hole
    return {
        "name": entry["name"],
        "D_m": entry["D_m"],
        "N_holes": entry["N_holes"],
        "L_m": entry["L_m"],
        "Cd_water": entry["Cd_water"],
        "Cd": entry["Cd_water"],                 
        "A_physical": A_physical,
        "L_over_D": entry["L_m"] / entry["D_m"],
        "complete": True,
    }


_ALL_INJECTORS = {k: _build_injector(v) for k, v in INJECTORS.items()}

if SELECTED_INJECTOR not in _ALL_INJECTORS:
    raise ValueError(f"SELECTED_INJECTOR='{SELECTED_INJECTOR}' not found in INJECTORS")

INJECTOR = _ALL_INJECTORS[SELECTED_INJECTOR]

if not INJECTOR["complete"]:
    missing = [k for k in ("D_m", "N_holes", "L_m", "Cd_water")
               if INJECTOR.get(k) is None]
    raise ValueError(
        f"Selected injector '{SELECTED_INJECTOR}' is incomplete; "
        f"missing fields: {missing}. Fill in the INJECTORS dict before running."
    )


# ============================================================
# OPERATING CONDITIONS (UPSTREAM)
# ============================================================
OPERATING = {
    "P1_psi": 600,                   # PT5 manifold avg
    "Pc_design_psi": 500,            # target chamber pressure
}


# ============================================================
# WEIGHT (KAPPA) - do not edit
# ============================================================
KAPPA_SWEEP = [0.0]
KAPPA_PRIMARY = 0.0


# ============================================================
# P2 (DOWNSTREAM PRESSURE) - do not edit
# ============================================================
SWEEP = {
    "P2_min_pa": 100000,
    "N_points": 200,
}


# ============================================================
# CALIBRATION
# ============================================================
CALIBRATION = {
    "kappa_min": 0.00001,
    "kappa_max": 0.18,
    "kappa_steps": 50,
    "mdot_measured_lb_s": 1.85,
    "burn_duration_s": 14,
    "total_ox_consumed_lb": 27,
    "pass_band_pct": 15, # percentage margin
}


# ============================================================
# PREDICTION TARGETS
# ============================================================
# Engine team supplies target_mdot_kgs to drive orifice-count iteration.
#
# Only ṁ is a meaningful target here: N changes total orifice area, not
# the per-hole discharge coefficient. Cd is set by edge condition and L/D,
# both of which are intrinsic to a single hole's geometry. To move Cd,
# change the chamfer/round, the hole diameter, or L — not N.
ITERATION = {
    "target_mdot_kgs": 1.20389,
    "fixed_diameter_m": None,             # None -> use normal injector D
    "apply_correction_factor": True,      # multiply Cd_water by HF2 factor
    "N_search_min": 30,
    "N_search_max": 200,
}


# ============================================================
# OUTPUT FILE NAMES
# ============================================================
OUTPUT_FILES = {
    "csv":               "mdot_vs_dP.csv",
    "plot":              "mdot_vs_dP.png",
    "metadata":          "run_metadata.json",
    "calibration_json":  "calibrated_params.json",
    "calibration_plot":  "calibration_kappa_sweep.png",
    "calibration_curves": "calibration_curves_at_best_kappa.png",
    "iteration_csv":     "iteration_results.csv",
    "iteration_plot":    "iteration_results.png",
}