import math

# ============================================================
# INJECTORS
# ============================================================
INJECTORS = {
    "hf2_fired": {
        "name": "HF2 INJECTOR (FIRED)",
        "D_m": 0.00119063,                    # hole diameter (m)
        "N_holes": 66,                        # number of orifices (count)
        "L_m": 0.00635,                       # hole depth / plate thickness (m)
        "Cd_water": 0.6411441306,             # discharge coefficient from water flow
    },

    "hf3_pilot": {
        "name": "HF2 UNFIRED INJ. (PILOT 1mm ORIFICE)", 
        "D_m": 0.001,                         # hole diameter (m)
        "N_holes": 36,                        # number of orifices (count)
        "L_m": 0.00635,                       # hole depth / plate thickness (m)
        "Cd_water": 0.6879905414,             # discharge coefficient from water flow
    },

     "hf3_drilled": {
        "name": "HF2 INJECTOR (NOT FIRED)",   # REAL HOLES THEYRE NOT PILOT 
        "D_m": 0.0012,                        # hole diameter (m)
        "N_holes": 36,                        # number of orifices (count)
        "L_m": 0.00635,                       # hole depth / plate thickness (m)
        "Cd_water": 0.6879905414,             # discharge coefficient from water flow
    },
}

SELECTED_INJECTOR = "hf3_drilled"       # CHOOSE THE INJECTOR


# ============================================================
# OPERATING CONDITIONS (UPSTREAM)
# ============================================================
OPERATING = {
    "P1_psi": 600,                   # Manifold (psi)
    "Pc_design_psi": 500,            # Chamber (psi)
}

# ============================================================
# ORIFICE PREDICTION TARGETS (prediction.py)
# ============================================================
ITERATION = {
    "target_mdot_kgs": 1.2040992992143,
    "fixed_diameter_m": None,             # None uses normal D_m
    "apply_correction_factor": True,      # multiply Cd_water by HF2 factor
    "N_search_min": 36,                   # minimum orifice count for search
    "N_search_max": 75,                   # maximum orifice count for search
}

# ============================================================
# MODEL CORRECTION (HF2 derived multiplier to gap 10% difference)
# ============================================================
EMPIRICAL_CORRECTIONS = {
    # calibrate.py overwrites this
    # this only goes to prediction.py
    "Cd_correction_factor_HF2": 0.9042847181526282,    # mdot_measured / mdot_predicted
}

# ============================================================
# WEIGHT (KAPPA) - do not edit
# ============================================================
KAPPA = 0.0   # locked by HF2 calibration

# ============================================================
# P2 (DOWNSTREAM PRESSURE) - do not edit
# ============================================================
SWEEP = {
    "P2_min_pa": 100000,    # lower bound of P2 sweep (Pa), ( atmospheric )
    "N_points": 1000,       # P2 sweep resolution (count)
}

# ============================================================
# OUTPUT FILE NAMES
# ============================================================
OUTPUT_FILES = {
    "csv":               "mdot_vs_dP.csv",
    "plot":              "mdot_vs_dP.png",
    "calibration_json":  "calibrated_params.json",
    "iteration_csv":     "iteration_results.csv",
    "iteration_plot":    "iteration_results.png",
}


# ============================================================
# DERIVED PROPERTIES (do not edit)
# ============================================================
def _build_injector(entry):
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