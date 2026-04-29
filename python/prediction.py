import os
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inputs
import model

# ============================================================
# CONSTANTS
# ============================================================
PSI_TO_PA = model.PSI_TO_PA
LB_TO_KG = model.LB_TO_KG
FLUID = model.FLUID
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
CALIBRATION_DIR = os.path.join(REPO_ROOT, "calibrations")

# ============================================================
# CORRECTION FACTOR RESOLUTION
# ============================================================
def resolve_correction_factor(apply_correction):
    if not apply_correction:
        return 1.0, "disabled (apply_correction_factor=False)"

    cal_path = os.path.join(
        CALIBRATION_DIR, inputs.OUTPUT_FILES["calibration_json"]
    )
    if os.path.exists(cal_path):
        try:
            with open(cal_path) as f:
                cal = json.load(f)
            factor = cal["empirical_corrections"]["Cd_correction_factor"]
            return float(factor), f"calibration JSON ({cal_path})"
        except (KeyError, json.JSONDecodeError):
            pass  

    corr = getattr(inputs, "EMPIRICAL_CORRECTIONS", None)
    if corr is None:
        raise ValueError(
            "apply_correction_factor=True but no correction available: "
            "neither calibration JSON nor inputs.EMPIRICAL_CORRECTIONS found."
        )
    return float(corr["Cd_correction_factor_HF2"]), "inputs.EMPIRICAL_CORRECTIONS"

# ============================================================
# INPUT VALIDATION
# ============================================================
def validate_iteration_inputs():
    it = inputs.ITERATION
    inj = inputs.INJECTOR

    if not inj["complete"]:
        raise ValueError(
            f"Selected injector '{inputs.SELECTED_INJECTOR}' is incomplete. "
            "Fill in INJECTORS dict before running prediction.py."
        )

    if it["N_search_min"] < 1:
        raise ValueError(f"N_search_min={it['N_search_min']} must be >= 1")
    if it["N_search_max"] <= it["N_search_min"]:
        raise ValueError(
            f"N_search_max={it['N_search_max']} must exceed "
            f"N_search_min={it['N_search_min']}"
        )

    D = it["fixed_diameter_m"] if it["fixed_diameter_m"] is not None else inj["D_m"]
    if D is None or D <= 0:
        raise ValueError(f"Fixed diameter must be positive, got D={D}")

    P1_psi = inputs.OPERATING["P1_psi"]
    Pc = inputs.OPERATING["Pc_design_psi"]
    if Pc >= P1_psi:
        raise ValueError(
            f"Pc_design_psi={Pc} >= P1_psi={P1_psi}; backflow regime."
        )

# ============================================================
# PER ORIFICE COUNT OPTIMIZATION
# ============================================================
def build_unit_curve(P1_pa, Pc_design_psi):
    result = model.run_model(
        Cd=1.0,
        A_physical=1.0,
        P1=P1_pa,
        kappa_list=[inputs.KAPPA],
        P2_min=inputs.SWEEP["P2_min_pa"],
        N_points=inputs.SWEEP["N_points"],
        Pc_design_psi=Pc_design_psi,
    )
    op = result["operating_summary"][inputs.KAPPA]
    return {
        "mdot_at_crit_per_CdA": op["mdot_at_crit_kgs"],
        "mdot_at_Pc_per_CdA":   op.get("mdot_at_design_Pc_kgs", float("nan")),
        "P2_crit_psi":          result["P2_crit_psi"],
        "is_choked_at_Pc":      Pc_design_psi < result["P2_crit_psi"],
    }

def predict_for_N(N, D, Cd, unit_curve):
    A_per_hole = math.pi / 4 * D ** 2
    A_physical = N * A_per_hole
    CdA = Cd * A_physical

    return {
        "N": int(N),
        "D_m": D,
        "A_per_hole_mm2": A_per_hole * 1e6,
        "A_physical_mm2": A_physical * 1e6,
        "mdot_at_Pc_kgs":   CdA * unit_curve["mdot_at_Pc_per_CdA"],
        "mdot_at_Pc_lbs":   CdA * unit_curve["mdot_at_Pc_per_CdA"] / LB_TO_KG,
        "mdot_at_crit_kgs": CdA * unit_curve["mdot_at_crit_per_CdA"],
        "mdot_at_crit_lbs": CdA * unit_curve["mdot_at_crit_per_CdA"] / LB_TO_KG,
        "P2_crit_psi":      unit_curve["P2_crit_psi"],
        "is_choked_at_Pc":  unit_curve["is_choked_at_Pc"],
    }

def sweep_N(N_array, D, Cd, unit_curve):
    return [predict_for_N(N, D, Cd, unit_curve) for N in N_array]

# ============================================================
# RECOMMENDATION
# ============================================================
def find_recommended_N(sweep, target_mdot_kgs):
    if target_mdot_kgs is None:
        return None
    for row in sweep:
        if row["mdot_at_Pc_kgs"] >= target_mdot_kgs:
            return row
    return None

# ============================================================
# OUTPUT
# ============================================================
def print_summary(sweep, D, Cd_water, factor, factor_source, Cd_effective,
                  target_mdot_kgs, recommended, current_row):
    inj = inputs.INJECTOR
    op = inputs.OPERATING

    print()
    print("=" * 72)
    print("ORIFICE COUNTS")
    print("=" * 72)

    print()
    print("CONFIGURATION")
    print("-" * 72)
    print(f"  Injector            {inj['name']}")
    print(f"  Diameter (fixed)    {D*1e3:.4f} mm  ({D*1e6:.1f} µm)")
    print(f"  Cd_water            {Cd_water:.4f}")
    print(f"  Correction factor   {factor:.4f}")
    print(f"  Correction source   {factor_source}")
    print(f"  Cd effective        {Cd_effective:.4f}  (Cd_water × factor)")
    print(f"  P₁                  {op['P1_psi']:.0f} psi")
    print(f"  Pc design           {op['Pc_design_psi']:.0f} psi")
    print(f"  ΔP at design        {op['P1_psi']-op['Pc_design_psi']:.0f} psi")
    print(f"  N search range      [{inputs.ITERATION['N_search_min']}, "
          f"{inputs.ITERATION['N_search_max']}]")
    if target_mdot_kgs is not None:
        print(f"  Target ṁ            {target_mdot_kgs:.4f} kg/s "
              f"({target_mdot_kgs/LB_TO_KG:.3f} lb/s)")
    else:
        print(f"  Target ṁ            (none specified — table only)")

    if current_row is not None:
        print()
        print("CURRENT STATE")
        print("-" * 72)
        print(f"  N                   {current_row['N']}")
        print(f"  A_physical          {current_row['A_physical_mm2']:.2f} mm²")
        print(f"  ṁ at Pc             {current_row['mdot_at_Pc_kgs']:.4f} kg/s "
              f"({current_row['mdot_at_Pc_lbs']:.3f} lb/s)")
        print(f"  ṁ at choke          {current_row['mdot_at_crit_kgs']:.4f} kg/s "
              f"({current_row['mdot_at_crit_lbs']:.3f} lb/s)")
        regime = "CHOKED" if current_row["is_choked_at_Pc"] else "sub-critical"
        print(f"  Regime at Pc        {regime}")

    print()
    print(f"N SWEEP ({len(sweep)} points)")
    print("-" * 72)
    print(f"  {'N':>5}{'A (mm²)':>11}{'ṁ at Pc (kg/s)':>18}"
          f"{'lb/s':>9}{'ṁ at choke':>13}{'lb/s':>9}{'regime':>14}")
    print(f"  {'-'*5:>5}{'-'*11:>11}{'-'*18:>18}{'-'*9:>9}"
          f"{'-'*13:>13}{'-'*9:>9}{'-'*14:>14}")

    n_rows = len(sweep)
    rows_to_show = set(range(n_rows))

    for i in sorted(rows_to_show):
        r = sweep[i]
        regime = "choked" if r["is_choked_at_Pc"] else "sub-crit"
        marker = ""
        if recommended is not None and r["N"] == recommended["N"]:
            marker = "  ← recommended"
        elif current_row is not None and r["N"] == current_row["N"]:
            marker = "  ← current"
        print(f"  {r['N']:>5}"
              f"{r['A_physical_mm2']:>11.2f}"
              f"{r['mdot_at_Pc_kgs']:>15.4f}"
              f"{r['mdot_at_Pc_lbs']:>11.3f}"
              f"{r['mdot_at_crit_kgs']:>13.4f}"
              f"{r['mdot_at_crit_lbs']:>9.3f}"
              f"{regime:>14}{marker}")

    print()
    print("RECOMMENDATION")
    print("-" * 72)
    if target_mdot_kgs is None:
        print("  No target_mdot_kgs specified. Pick N from the table above")
        print("  based on engine spec.")
    elif recommended is None:
        max_row = max(sweep, key=lambda r: r["mdot_at_Pc_kgs"])
        print(f"  ⚠ No N in [{sweep[0]['N']}, {sweep[-1]['N']}] reaches target "
              f"{target_mdot_kgs:.4f} kg/s.")
        print(f"    Max in sweep: N={max_row['N']} → "
              f"{max_row['mdot_at_Pc_kgs']:.4f} kg/s. "
              f"Widen N_search_max or revisit D.")
    else:
        print(f"  Smallest N meeting target: N = {recommended['N']}")
        print(f"  Predicted ṁ at Pc:   {recommended['mdot_at_Pc_kgs']:.4f} kg/s "
              f"({recommended['mdot_at_Pc_lbs']:.3f} lb/s)")
        print(f"  Margin over target:  "
              f"{(recommended['mdot_at_Pc_kgs']-target_mdot_kgs)*1000:.1f} g/s "
              f"({(recommended['mdot_at_Pc_kgs']/target_mdot_kgs-1)*100:+.1f}%)")
        if current_row is not None:
            delta = recommended["N"] - current_row["N"]
            direction = "more" if delta > 0 else ("fewer" if delta < 0 else "same")
            print(f"  vs current N={current_row['N']}: "
                  f"{abs(delta)} {direction} hole(s)")
        print()

def save_csv(sweep, path):
    df = pd.DataFrame(sweep)
    df.to_csv(path, index=False)
    print(f"  CSV:  {path}")


def save_plot(sweep, target_mdot_kgs, recommended, current_row,
              D, Cd_effective, path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.facecolor": "#fafafa",
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))

    Ns = [r["N"] for r in sweep]
    mdots_pc = [r["mdot_at_Pc_kgs"] for r in sweep]
    mdots_crit = [r["mdot_at_crit_kgs"] for r in sweep]

    ax.plot(Ns, mdots_pc, "-", color="#224a8b", linewidth=2.4,
            label="ṁ at design Pc")
    ax.plot(Ns, mdots_crit, "--", color="#C26A06", linewidth=1.4, alpha=0.7,
            label="ṁ at choke (HEM crit)")

    if target_mdot_kgs is not None:
        ax.axhline(target_mdot_kgs, color="#2ca02c", linestyle="--",
                   linewidth=1.6, alpha=0.8,
                   label=f"Target ({target_mdot_kgs:.4f} kg/s)")

    if recommended is not None:
        ax.axvline(recommended["N"], color="#2ca02c", linestyle=":",
                   alpha=0.5, linewidth=1.2)
        ax.scatter([recommended["N"]], [recommended["mdot_at_Pc_kgs"]],
                   s=140, marker="*", color="#2ca02c",
                   edgecolor="#1f5e1f", linewidth=1.3, zorder=10,
                   label=f"Recommended N={recommended['N']}")

    if current_row is not None:
        ax.scatter([current_row["N"]], [current_row["mdot_at_Pc_kgs"]],
                   s=110, marker="o", color="#d62728",
                   edgecolor="#8b0000", linewidth=1.3, zorder=9,
                   label=f"Current N={current_row['N']}")

    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin / LB_TO_KG, ymax / LB_TO_KG)
    ax2.set_ylabel("ṁ (lb/s)", color="#555555")
    ax2.tick_params(axis="y", colors="#555555")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax.set_xlabel("N (number of orifices)")
    ax.set_ylabel("ṁ (kg/s)")

    inj = inputs.INJECTOR
    op = inputs.OPERATING
    fig.suptitle(
        f"Orifice-count iteration — {inj['name']}",
        fontsize=14, fontweight="bold", y=0.97,
    )
    ax.set_title(
        f"D={D*1e3:.4f} mm · Cd_eff={Cd_effective:.4f} · "
        f"P₁={op['P1_psi']:.0f} psi · Pc={op['Pc_design_psi']:.0f} psi",
        fontsize=11, color="#555555", pad=10,
    )

    legend = ax.legend(loc="lower right", framealpha=0.85,
                       edgecolor="#cccccc", fontsize=10)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")

# ============================================================
# RUN AS SCRIPT
# ============================================================
if __name__ == "__main__":
    print()
    print("Validating iteration inputs...")
    validate_iteration_inputs()

    it = inputs.ITERATION
    inj = inputs.INJECTOR

    D = it["fixed_diameter_m"] if it["fixed_diameter_m"] is not None else inj["D_m"]
    Cd_water = inj["Cd_water"]
    factor, factor_source = resolve_correction_factor(it["apply_correction_factor"])
    Cd_effective = Cd_water * factor

    P1_pa = inputs.OPERATING["P1_psi"] * PSI_TO_PA
    Pc_psi = inputs.OPERATING["Pc_design_psi"]

    print("Building unit P2 curve (one model run)...")
    unit_curve = build_unit_curve(P1_pa, Pc_psi)

    N_array = np.arange(it["N_search_min"], it["N_search_max"] + 1)
    print(f"Sweeping N from {it['N_search_min']} to {it['N_search_max']} "
          f"({len(N_array)} points)...")
    sweep = sweep_N(N_array, D, Cd_effective, unit_curve)

    current_row = None
    if inj["N_holes"] is not None and abs(inj["D_m"] - D) < 1e-9:
        current_row = predict_for_N(inj["N_holes"], D, Cd_effective, unit_curve)

    target = it["target_mdot_kgs"]
    recommended = find_recommended_N(sweep, target)

    print_summary(sweep, D, Cd_water, factor, factor_source, Cd_effective,
                  target, recommended, current_row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Outputs")
    print("-" * 72)
    save_csv(sweep, os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["iteration_csv"]))
    save_plot(sweep, target, recommended, current_row, D, Cd_effective,
              os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["iteration_plot"]))
    print()
    print("Done.")
    print()