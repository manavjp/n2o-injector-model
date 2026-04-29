import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import inputs
import model


# ============================================================
# CONSTANTS
# ============================================================
PSI_TO_PA = model.PSI_TO_PA
LB_TO_KG = model.LB_TO_KG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
CALIBRATION_DIR = os.path.join(REPO_ROOT, "calibrations")


# ============================================================
# CALIBRATION CORE
# ============================================================
def run_calibration(Cd, A_physical, P1, kappa_array, P2_min, N_points,
                    mdot_measured_kgs):
    """Sweep kappa, run model at each, compute error vs measured."""
    sweep_results = []

    for kappa in kappa_array:
        result = model.run_model(
            Cd=Cd,
            A_physical=A_physical,
            P1=P1,
            kappa_list=[kappa],
            P2_min=P2_min,
            N_points=N_points,
        )
        op = result["operating_summary"][kappa]
        predicted = op["mdot_at_crit_kgs"]
        error_pct = (predicted - mdot_measured_kgs) / mdot_measured_kgs * 100.0

        sweep_results.append({
            "kappa": float(kappa),
            "mdot_predicted_kgs": float(predicted),
            "mdot_predicted_lbs": float(predicted / LB_TO_KG),
            "error_pct": float(error_pct),
            "P2_crit_psi": float(result["P2_crit_psi"]),
            "P2_crit_over_Psat": float(result["P2_crit_over_Psat"]),
            "mdot_crit_HEM_kgs": float(result["mdot_crit_HEM_kgs"]),
        })

    return sweep_results


def find_best_kappa(sweep_results, pass_band_pct):
    """Identify best-fit kappa and pass band."""
    abs_errors = [abs(r["error_pct"]) for r in sweep_results]
    best_idx = int(np.argmin(abs_errors))
    best = sweep_results[best_idx]

    passing = [r for r in sweep_results
               if abs(r["error_pct"]) <= pass_band_pct]
    if passing:
        pass_band = (
            min(r["kappa"] for r in passing),
            max(r["kappa"] for r in passing),
        )
    else:
        pass_band = None

    return best, pass_band, passing


# ============================================================
# OUTPUT
# ============================================================
def print_summary(sweep_results, best, pass_band, passing,
                  mdot_measured_kgs, pass_band_pct):
    print()
    print("=" * 72)
    print("N2O INJECTOR MODEL CALIBRATION (single-fire)")
    print("=" * 72)

    inj = inputs.INJECTOR
    op = inputs.OPERATING
    cal = inputs.CALIBRATION

    print()
    print("CONFIGURATION")
    print("-" * 72)
    print(f"  Injector            {inj['name']}")
    print(f"  Cd                  {inj['Cd']:.4f}")
    print(f"  A_physical          {inj['A_physical']*1e6:.2f} mm²")
    print(f"  L/D                 {inj['L_over_D']:.2f}")
    print(f"  P₁ (manifold)       {op['P1_psi']:.0f} psi")

    print()
    print("HOT FIRE REFERENCE")
    print("-" * 72)
    print(f"  Measured ṁ          {cal['mdot_measured_lb_s']:.3f} lb/s "
          f"({mdot_measured_kgs:.4f} kg/s)")
    print(f"  Burn duration       {cal['burn_duration_s']} s (liquid phase)")
    print(f"  Total ox consumed   {cal['total_ox_consumed_lb']} lb")
    print(f"  Pass band           ±{pass_band_pct}% per Waxman §IV.D")

    print()
    print(f"KAPPA SWEEP ({len(sweep_results)} points)")
    print("-" * 72)
    print(f"  {'κ':<8}{'ṁ pred':>14}{'lb/s':>10}{'error':>11}{'P2/Psat':>11}  status")
    print(f"  {'-'*8:<8}{'-'*14:>14}{'-'*10:>10}{'-'*11:>11}{'-'*11:>11}")

    # Print every Nth row to avoid wall of text, plus best
    n = len(sweep_results)
    stride = max(1, n // 15)
    rows_to_print = set(range(0, n, stride))
    rows_to_print.add(0)
    rows_to_print.add(n - 1)
    rows_to_print.add(sweep_results.index(best))

    for i in sorted(rows_to_print):
        r = sweep_results[i]
        status = "PASS" if abs(r["error_pct"]) <= pass_band_pct else "FAIL"
        marker = "  ← best" if r["kappa"] == best["kappa"] else ""
        print(f"  {r['kappa']:<8.4f}"
              f"{r['mdot_predicted_kgs']:>11.4f} kg/s"
              f"{r['mdot_predicted_lbs']:>10.3f}"
              f"{r['error_pct']:>+10.2f}%"
              f"{r['P2_crit_over_Psat']:>11.3f}  {status}{marker}")

    print()
    print("BEST-FIT RESULT")
    print("-" * 72)
    print(f"  Best κ              {best['kappa']:.4f}")
    print(f"  Predicted ṁ         {best['mdot_predicted_kgs']:.4f} kg/s "
          f"({best['mdot_predicted_lbs']:.3f} lb/s)")
    print(f"  Error vs measured   {best['error_pct']:+.2f}%")
    print(f"  P2_crit / P_sat     {best['P2_crit_over_Psat']:.3f}  (Waxman ~0.80 expected)")

    if pass_band:
        print(f"  Pass band κ range   [{pass_band[0]:.4f}, "
              f"{pass_band[1]:.4f}]  ({len(passing)} values within ±{pass_band_pct}%)")
    else:
        print(f"  ⚠ No κ within ±{pass_band_pct}% — model cannot match this fire")
        print(f"    Closest is κ = {best['kappa']:.4f} at {best['error_pct']:+.2f}%")

    factor = mdot_measured_kgs / best["mdot_predicted_kgs"]
    Cd_2phase = inj["Cd_water"] * factor
    print()
    print("DERIVED 2-PHASE CD")
    print("-" * 72)
    print(f"  Cd correction factor   {factor:.4f}")
    print(f"  Cd_2phase              {Cd_2phase:.4f}  "
          f"(= Cd_water {inj['Cd_water']:.4f} × {factor:.4f})")

    if best["kappa"] == sweep_results[0]["kappa"]:
        print()
        print(f"  ⚠ Best κ is at LOWER bound of sweep ({best['kappa']:.4f}).")
        print(f"    Treat as 'pure HEM' regime.")
    elif best["kappa"] == sweep_results[-1]["kappa"]:
        print()
        print(f"  ⚠ Best κ is at UPPER bound of sweep ({best['kappa']:.4f}).")
        print(f"    True best may be even higher. Try widening kappa_max")

    print()
    print("VALIDATION CHANNELS")
    print("-" * 72)
    ch1 = abs(best["error_pct"]) <= pass_band_pct
    print(f"  {'✓ PASS' if ch1 else '✗ FAIL'}  Channel 1 — Average ṁ")
    print(f"           predicted {best['mdot_predicted_kgs']:.4f} kg/s, "
          f"measured {mdot_measured_kgs:.4f} kg/s")

    ratio = best["P2_crit_over_Psat"]
    ch2 = 0.7 <= ratio <= 0.9
    print(f"  {'✓ PASS' if ch2 else '⚠ CHECK'}  Channel 2 — Choke ratio P2/P_sat")
    print(f"           predicted {ratio:.3f}, Waxman expected ~0.80")


def save_calibration_json(sweep_results, best, pass_band,
                          mdot_measured_kgs, pass_band_pct, path):
    inj = inputs.INJECTOR
    op = inputs.OPERATING
    cal = inputs.CALIBRATION

    out = {
        "timestamp": datetime.now().isoformat(),
        "calibration_id": f"{inj['name']}_calibration",
        "single_fire_calibration": True,
        "preliminary": True,
        "injector": dict(inj),
        "operating": dict(op),
        "hot_fire_reference": {
            "mdot_measured_kgs": float(mdot_measured_kgs),
            "mdot_measured_lbs": float(cal["mdot_measured_lb_s"]),
            "burn_duration_s": cal["burn_duration_s"],
            "total_ox_consumed_lb": cal["total_ox_consumed_lb"],
        },
        "calibration_config": {
            "kappa_min": cal["kappa_min"],
            "kappa_max": cal["kappa_max"],
            "kappa_steps": cal["kappa_steps"],
            "pass_band_pct": pass_band_pct,
        },
        "best_fit": dict(best),
        "empirical_corrections": {
            "Cd_correction_factor": float(
                mdot_measured_kgs / best["mdot_predicted_kgs"]
            ),
            "Cd_2phase_calibrated": float(
                inj["Cd_water"]
                * (mdot_measured_kgs / best["mdot_predicted_kgs"])
            ),
            "interpretation": (
                "Multiplicative correction on water-derived Cd to match "
                "measured two-phase N2O ṁ at this fire's operating point. "
                "Apply forward to similar-geometry injectors via "
                "predict_iteration.py."
            ),
            "single_fire_basis": True,
        },
        "pass_band_kappa_range": (
            list(pass_band) if pass_band else None
        ),
        "validation_status": {
            "channel_1_avg_mdot": abs(best["error_pct"]) <= pass_band_pct,
            "channel_2_choke_ratio": 0.7 <= best["P2_crit_over_Psat"] <= 0.9,
        },
        "applicability": {
            "l_over_d_at_calibration": inj["L_over_D"],
            "recommended_l_over_d_range": [
                inj["L_over_D"] * 0.8,
                inj["L_over_D"] * 1.2,
            ],
            "note": (
                "Single-fire calibration. Apply only to injectors with "
                "similar L/D and edge condition. Recalibrate when geometry "
                "regime changes."
            ),
        },
        "sweep_results": sweep_results,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Calibration JSON: {path}")


def save_kappa_sweep_plot(sweep_results, best, mdot_measured_kgs,
                          pass_band_pct, path):
    fig, ax = plt.subplots(figsize=(11, 7))

    kappas = [r["kappa"] for r in sweep_results]
    mdots = [r["mdot_predicted_kgs"] for r in sweep_results]

    band_low = mdot_measured_kgs * (1 - pass_band_pct / 100)
    band_high = mdot_measured_kgs * (1 + pass_band_pct / 100)
    ax.axhspan(band_low, band_high, color="lightgreen", alpha=0.3,
               label=f"±{pass_band_pct}% pass band")

    ax.axhline(mdot_measured_kgs, color="green", linestyle="--",
               linewidth=1.5,
               label=f"Measured ({mdot_measured_kgs:.4f} kg/s)")

    ax.plot(kappas, mdots, "-", color="C0", linewidth=1.8,
            label="Predicted ṁ at critical")

    ax.plot(best["kappa"], best["mdot_predicted_kgs"],
            "*", color="red", markersize=18, zorder=5,
            label=f"Best κ = {best['kappa']:.4f}")
    ax.axvline(best["kappa"], color="red", linestyle=":", alpha=0.4)

    inj = inputs.INJECTOR
    ax.set_xlabel("κ (NHNE weighting)")
    ax.set_ylabel("Predicted ṁ at critical (kg/s)")
    ax.set_title(
        f"Calibration κ Sweep — {inj['name']} (single-fire)\n"
        f"L/D={inj['L_over_D']:.2f}, Cd={inj['Cd']:.4f}, "
        f"P₁={inputs.OPERATING['P1_psi']:.0f} psi"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  κ sweep plot:     {path}")


def save_curves_at_best_kappa(best, mdot_measured_kgs, path):
    P1_pa = inputs.OPERATING["P1_psi"] * PSI_TO_PA

    result = model.run_model(
        Cd=inputs.INJECTOR["Cd"],
        A_physical=inputs.INJECTOR["A_physical"],
        P1=P1_pa,
        kappa_list=[best["kappa"]],
        P2_min=inputs.SWEEP["P2_min_pa"],
        N_points=inputs.SWEEP["N_points"],
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    dp_psi = result["dP_array"] / PSI_TO_PA

    ax.plot(dp_psi, result["mdot_spi"], "--", color="lightgray",
            linewidth=1.4, label="SPI alone")
    ax.plot(dp_psi, result["mdot_hem"], ":", color="darkgray",
            linewidth=1.4, label="HEM (clamped)")
    nhne = result["nhne_curves"][best["kappa"]]
    ax.plot(dp_psi, nhne, "-", color="C0", linewidth=2.4,
            label=f"NHNE at κ={best['kappa']:.4f}")

    ax.axhline(mdot_measured_kgs, color="green", linestyle="--",
               linewidth=1.5,
               label=f"Measured ({mdot_measured_kgs:.4f} kg/s)")

    ax.axvline(result["dP_crit_psi"], color="red", linestyle=":", alpha=0.5)
    ax.plot(result["dP_crit_psi"], result["mdot_crit_HEM_kgs"],
            "*", color="red", markersize=14, zorder=5,
            label=f"HEM crit ({result['mdot_crit_HEM_kgs']:.4f} kg/s)")

    inj = inputs.INJECTOR
    ax.set_xlabel("ΔP across injector (psi)")
    ax.set_ylabel("ṁ (kg/s)")
    ax.set_title(
        f"Calibrated NHNE — {inj['name']}\n"
        f"Cd={inj['Cd']:.3f}, L/D={inj['L_over_D']:.2f}, "
        f"κ={best['kappa']:.4f}, P₁={inputs.OPERATING['P1_psi']:.0f} psi"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Calibrated curve: {path}")


# ============================================================
# RUN AS SCRIPT
# ============================================================
if __name__ == "__main__":
    print()
    print("Validating inputs...")
    model.validate_inputs()

    cal = inputs.CALIBRATION
    pass_band_pct = cal["pass_band_pct"]
    mdot_measured_kgs = cal["mdot_measured_lb_s"] * LB_TO_KG

    P1_pa = inputs.OPERATING["P1_psi"] * PSI_TO_PA
    kappa_array = np.linspace(
        cal["kappa_min"], cal["kappa_max"], cal["kappa_steps"]
    )

    print(f"Sweeping κ from {cal['kappa_min']} to {cal['kappa_max']} "
          f"in {cal['kappa_steps']} steps...")

    sweep_results = run_calibration(
        Cd=inputs.INJECTOR["Cd"],
        A_physical=inputs.INJECTOR["A_physical"],
        P1=P1_pa,
        kappa_array=kappa_array,
        P2_min=inputs.SWEEP["P2_min_pa"],
        N_points=inputs.SWEEP["N_points"],
        mdot_measured_kgs=mdot_measured_kgs,
    )

    best, pass_band, passing = find_best_kappa(sweep_results, pass_band_pct)

    print_summary(sweep_results, best, pass_band, passing,
                  mdot_measured_kgs, pass_band_pct)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    print("Outputs")
    print("-" * 72)
    save_calibration_json(
        sweep_results, best, pass_band, mdot_measured_kgs, pass_band_pct,
        os.path.join(CALIBRATION_DIR, inputs.OUTPUT_FILES["calibration_json"]),
    )
    save_kappa_sweep_plot(
        sweep_results, best, mdot_measured_kgs, pass_band_pct,
        os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["calibration_plot"]),
    )
    save_curves_at_best_kappa(
        best, mdot_measured_kgs,
        os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["calibration_curves"]),
    )
    print()
    print("Done.")
    print()