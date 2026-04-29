import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

import inputs


# ============================================================
# CONSTANTS
# ============================================================
FLUID = "NitrousOxide"
PSI_TO_PA = 6894.76
LB_TO_KG = 0.453592

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


# ============================================================
# INPUT VALIDATION
# ============================================================
def validate_inputs():
    errors, warnings = [], []
    Cd = inputs.INJECTOR["Cd"]
    A = inputs.INJECTOR["A_physical"]
    P1_psi = inputs.OPERATING["P1_psi"]
    kappa_list = inputs.KAPPA_SWEEP

    if not (0 < Cd <= 1.0):
        errors.append(f"Cd={Cd} outside (0, 1]")
    if not (0.55 <= Cd <= 0.75):
        warnings.append(f"Cd={Cd:.3f} outside typical sharp-edged range (0.6–0.7)")
    if A <= 0:
        errors.append(f"A_physical={A} must be positive")
    if P1_psi <= 14.7:
        errors.append(f"P1_psi={P1_psi} must be above ambient")
    if any(k < 0 for k in kappa_list):
        errors.append(f"All kappa values must be non-negative: {kappa_list}")

    Pc = inputs.OPERATING.get("Pc_design_psi")
    if Pc is not None:
        if Pc <= 0:
            errors.append(f"Pc_design_psi={Pc} must be positive")
        if Pc >= P1_psi:
            warnings.append(
                f"Pc_design_psi={Pc} >= P1={P1_psi}; backflow regime, no flow possible"
            )

    if errors:
        for e in errors:
            print(f"  ✗ ERROR: {e}")
        raise ValueError("Invalid inputs; aborting.")

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")


# ============================================================
# THERMODYNAMIC STATE
# ============================================================
def saturated_upstream(P1):
    return {
        "T1":    PropsSI("T", "P", P1, "Q", 0, FLUID),
        "rho_L": PropsSI("D", "P", P1, "Q", 0, FLUID),
        "h1":    PropsSI("H", "P", P1, "Q", 0, FLUID),
        "s1":    PropsSI("S", "P", P1, "Q", 0, FLUID),
        "P_sat": P1,
    }


def downstream_isentropic(P2, s1):
    return {
        "h2":   PropsSI("H", "P", P2, "S", s1, FLUID),
        "rho2": PropsSI("D", "P", P2, "S", s1, FLUID),
        "x2":   PropsSI("Q", "P", P2, "S", s1, FLUID),
    }


# ============================================================
# FLOW MODELS
# ============================================================
def mdot_SPI(Cd, A, rho_L, dP):
    return Cd * A * np.sqrt(2 * rho_L * dP)


def mdot_HEM(Cd, A, rho2, h1, h2):
    return Cd * A * rho2 * np.sqrt(2 * (h1 - h2))


def mdot_NHNE(m_spi, m_hem, kappa):
    w_spi = kappa / (1 + kappa)
    w_hem = 1.0 / (1 + kappa)
    return w_spi * m_spi + w_hem * m_hem


# ============================================================
# CHOKE DETECTION
# ============================================================
def find_critical_point(mdot_hem_curve, P2_array):
    """Critical point = maximum of HEM curve."""
    idx = int(np.argmax(mdot_hem_curve))
    return idx, P2_array[idx], mdot_hem_curve[idx]


def clamp_hem_subcritical(mdot_hem_curve, crit_idx, mdot_crit, P2_array):
    """HEM clamped at critical for P2 < P2_crit."""
    clamped = mdot_hem_curve.copy()
    clamped[:crit_idx] = mdot_crit
    return clamped


# ============================================================
# MAIN MODEL RUN
# ============================================================
def run_model(Cd, A_physical, P1, kappa_list, P2_min, N_points,
              Pc_design_psi=None):
    upstream = saturated_upstream(P1)
    P_sat = upstream["P_sat"]

    P2_max = P_sat * 0.99
    P2_array = np.linspace(P2_min, P2_max, N_points)
    dP_array = P1 - P2_array

    mdot_spi = np.zeros(N_points)
    mdot_hem_raw = np.zeros(N_points)

    for i, P2 in enumerate(P2_array):
        ds = downstream_isentropic(P2, upstream["s1"])
        mdot_spi[i] = mdot_SPI(Cd, A_physical, upstream["rho_L"], dP_array[i])
        mdot_hem_raw[i] = mdot_HEM(
            Cd, A_physical, ds["rho2"], upstream["h1"], ds["h2"]
        )

    # critical point from HEM peak
    crit_idx, P2_crit, mdot_crit = find_critical_point(mdot_hem_raw, P2_array)

    # clamp HEM in subcritical
    mdot_hem = clamp_hem_subcritical(mdot_hem_raw, crit_idx, mdot_crit, P2_array)

    # (Pc from inputs)
    if Pc_design_psi is not None:
        Pc_design_pa = Pc_design_psi * PSI_TO_PA
        idx_design = int(np.argmin(np.abs(P2_array - Pc_design_pa)))
    else:
        idx_design = None

    # NHNE per kappa (uses clamped HEM)
    nhne_curves = {}
    operating_summary = {}
    for kappa in kappa_list:
        nhne = mdot_NHNE(mdot_spi, mdot_hem, kappa)
        nhne_curves[kappa] = nhne
        summary = {
            "kappa": float(kappa),
            "mdot_at_crit_kgs": float(nhne[crit_idx]),
            "mdot_at_crit_lbs": float(nhne[crit_idx] / LB_TO_KG),
            "mdot_at_ambient_kgs": float(nhne[0]),
            "mdot_at_ambient_lbs": float(nhne[0] / LB_TO_KG),
        }
        if idx_design is not None:
            summary["mdot_at_design_Pc_kgs"] = float(nhne[idx_design])
            summary["mdot_at_design_Pc_lbs"] = float(nhne[idx_design] / LB_TO_KG)
        operating_summary[kappa] = summary

    return {
        "upstream": upstream,
        "P2_array": P2_array,
        "dP_array": dP_array,
        "mdot_spi": mdot_spi,
        "mdot_hem_raw": mdot_hem_raw,
        "mdot_hem": mdot_hem,
        "nhne_curves": nhne_curves,
        "crit_idx": crit_idx,
        "P2_crit_pa": float(P2_crit),
        "P2_crit_psi": float(P2_crit / PSI_TO_PA),
        "dP_crit_pa": float(P1 - P2_crit),
        "dP_crit_psi": float((P1 - P2_crit) / PSI_TO_PA),
        "P2_crit_over_Psat": float(P2_crit / P_sat),
        "mdot_crit_HEM_kgs": float(mdot_crit),
        "operating_summary": operating_summary,
        "Pc_design_psi": Pc_design_psi,
        "idx_design": idx_design,
    }


# ============================================================
# SANITY CHECKS
# ============================================================
def sanity_checks(results, kappa_primary):
    checks = []
    primary = results["operating_summary"][kappa_primary]
    crit_idx = results["crit_idx"]

    spi_at_crit = float(results["mdot_spi"][crit_idx])
    hem_at_crit = float(results["mdot_hem"][crit_idx])
    if hem_at_crit <= primary["mdot_at_crit_kgs"] <= spi_at_crit:
        checks.append(("NHNE within SPI/HEM bounds at critical", "PASS"))
    else:
        checks.append(("NHNE within SPI/HEM bounds at critical", "FAIL"))

    ratio = results["P2_crit_over_Psat"]
    if 0.7 <= ratio <= 0.9:
        checks.append((f"Critical P2/Psat={ratio:.3f}", "PASS"))
    else:
        checks.append((f"Critical P2/Psat={ratio:.3f}", "OUTSIDE 0.7–0.9"))

    return checks


# ============================================================
# OUTPUTS
# ============================================================
def print_summary(results, kappa_primary):
    print()
    print("=" * 72)
    print("N2O INJECTOR MASS FLOW MODEL")
    print("=" * 72)
    inj = inputs.INJECTOR
    op = inputs.OPERATING
    upstream = results["upstream"]

    print()
    print("INPUTS")
    print("-" * 72)
    print(f"  Injector            {inj['name']}")
    print(f"  Cd                  {inj['Cd']:.4f}")
    print(f"  A_physical          {inj['A_physical']*1e6:.2f} mm²")
    print(f"  L/D                 {inj['L_over_D']:.2f}")
    print(f"  κ sweep             {inputs.KAPPA_SWEEP}")
    print(f"  κ primary           {kappa_primary}")
    print(f"  P₁ (manifold)       {op['P1_psi']:.0f} psi  ({op['P1_psi']*PSI_TO_PA/1e6:.3f} MPa)")
    print(f"  T₁ (sat)            {upstream['T1']:.2f} K  ({upstream['T1']-273.15:.1f} °C)")
    print(f"  ρ_L (sat)           {upstream['rho_L']:.1f} kg/m³")

    corr = getattr(inputs, "EMPIRICAL_CORRECTIONS", None)
    if corr is not None and inj.get("Cd_water") is not None:
        factor = corr["Cd_correction_factor_HF2"]
        Cd_water = inj["Cd_water"]
        Cd_2phase_ref = Cd_water * factor
        print()
        print("CD REFERENCE (model uses Cd_water; not applied to curves)")
        print("-" * 72)
        print(f"  Cd_water                {Cd_water:.4f}")
        print(f"  Cd correction factor    {factor:.4f}  (HF2-derived)")
        print(f"  Cd_2phase (reference)   {Cd_2phase_ref:.4f}")
        print(f"  Note: preliminary, single-fire derived. "
              f"Multiply ṁ by {factor:.3f} to scale.")

    print()
    print("CRITICAL POINT (from HEM peak)")
    print("-" * 72)
    print(f"  P₂_crit             {results['P2_crit_psi']:.1f} psi")
    print(f"  ΔP_crit             {results['dP_crit_psi']:.1f} psi")
    print(f"  P₂_crit / P_sat     {results['P2_crit_over_Psat']:.3f}")
    print(f"  ṁ_crit_HEM          {results['mdot_crit_HEM_kgs']:.4f} kg/s  "
          f"({results['mdot_crit_HEM_kgs']/LB_TO_KG:.3f} lb/s)")

    # Design operating point block
    if results["Pc_design_psi"] is not None:
        Pc = results["Pc_design_psi"]
        dP_design = op["P1_psi"] - Pc
        is_choked = Pc < results["P2_crit_psi"]
        print()
        print("DESIGN OPERATING POINT")
        print("-" * 72)
        print(f"  Pc design           {Pc:.0f} psi  ({Pc*PSI_TO_PA/1e6:.3f} MPa)")
        print(f"  ΔP at Pc design     {dP_design:.0f} psi")
        regime = "CHOKED" if is_choked else "sub-critical"
        print(f"  Operating regime    {regime}")
        if is_choked:
            print(f"  → ṁ insensitive to Pc; predicted ṁ ≈ ṁ_crit_HEM")
        else:
            print(f"  → ṁ depends on Pc; see per-κ values below")

    print()
    if results["Pc_design_psi"] is not None:
        print("OPERATING ṁ BY κ (at critical / at design Pc / at ambient)")
        print("-" * 72)
        print(f"  {'κ':<6}{'ṁ at crit':>14}{'lb/s':>10}"
              f"{'ṁ at Pc':>14}{'lb/s':>10}"
              f"{'ṁ at amb':>14}{'lb/s':>10}")
        print(f"  {'-'*6:<6}{'-'*14:>14}{'-'*10:>10}"
              f"{'-'*14:>14}{'-'*10:>10}"
              f"{'-'*14:>14}{'-'*10:>10}")
        for kappa, c in results["operating_summary"].items():
            marker = "  ← primary" if kappa == kappa_primary else ""
            print(f"  {kappa:<6.2f}"
                  f"{c['mdot_at_crit_kgs']:>11.4f} kg/s"
                  f"{c['mdot_at_crit_lbs']:>10.3f}"
                  f"{c['mdot_at_design_Pc_kgs']:>11.4f} kg/s"
                  f"{c['mdot_at_design_Pc_lbs']:>10.3f}"
                  f"{c['mdot_at_ambient_kgs']:>11.4f} kg/s"
                  f"{c['mdot_at_ambient_lbs']:>10.3f}{marker}")
    else:
        print("OPERATING ṁ BY κ (at critical / at ambient)")
        print("-" * 72)
        print(f"  {'κ':<6}{'ṁ at crit':>14}{'lb/s':>10}{'ṁ at ambient':>16}{'lb/s':>10}")
        print(f"  {'-'*6:<6}{'-'*14:>14}{'-'*10:>10}{'-'*16:>16}{'-'*10:>10}")
        for kappa, c in results["operating_summary"].items():
            marker = "  ← primary" if kappa == kappa_primary else ""
            print(f"  {kappa:<6.2f}"
                  f"{c['mdot_at_crit_kgs']:>11.4f} kg/s"
                  f"{c['mdot_at_crit_lbs']:>10.3f}"
                  f"{c['mdot_at_ambient_kgs']:>13.4f} kg/s"
                  f"{c['mdot_at_ambient_lbs']:>10.3f}{marker}")

    print()
    print("SANITY CHECKS")
    print("-" * 72)
    for label, status in sanity_checks(results, kappa_primary):
        marker = "✓" if "PASS" in status else "⚠"
        print(f"  {marker} {label:<48} {status}")
    print()


def save_csv(results, path):
    df = pd.DataFrame({
        "P2_psi": results["P2_array"] / PSI_TO_PA,
        "dP_psi": results["dP_array"] / PSI_TO_PA,
        "P2_Pa": results["P2_array"],
        "dP_Pa": results["dP_array"],
        "mdot_SPI_kg_s": results["mdot_spi"],
        "mdot_HEM_raw_kg_s": results["mdot_hem_raw"],
        "mdot_HEM_clamped_kg_s": results["mdot_hem"],
    })
    for kappa, curve in results["nhne_curves"].items():
        df[f"mdot_NHNE_k{kappa:.2f}_kg_s"] = curve
    df.to_csv(path, index=False)
    print(f"  CSV:      {path}")


def save_plot(results, kappa_primary, path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.facecolor": "#fafafa",
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))

    dp_psi = results["dP_array"] / PSI_TO_PA

    ax.plot(dp_psi, results["mdot_spi"], "--",
            color="#224a8b", linewidth=1.0, alpha=0.7,
            label="1 PHASE (SPI)")
    ax.plot(dp_psi, results["mdot_hem"], "--",
            color="#C26A06", linewidth=1.0, alpha=0.7,
            label="2 PHASE (HEM)")

    cmap = plt.cm.viridis
    n_kappa = len(results["nhne_curves"])
    for i, (kappa, curve) in enumerate(results["nhne_curves"].items()):
        is_primary = (kappa == kappa_primary)
        color = cmap(0.15 + 0.7 * i / max(n_kappa - 1, 1))
        ax.plot(
            dp_psi, curve,
            color=color,
            linewidth=2.8 if is_primary else 1.3,
            alpha=1.0 if is_primary else 0.55,
            label=f"NHNE κ={kappa:.2f}" + ("  ← primary" if is_primary else ""),
            zorder=3 if is_primary else 2,
        )

    ax.axvline(results["dP_crit_psi"], color="#d62728",
               linestyle=":", alpha=0.4, linewidth=1.2)
    ax.scatter(
        [results["dP_crit_psi"]],
        [results["mdot_crit_HEM_kgs"]],
        s=120, marker="D", color="#d62728",
        edgecolor="#8b0000", linewidth=1.5, zorder=10,
        label=f"Critical ({results['mdot_crit_HEM_kgs']:.3f} kg/s "
              f"@ ΔP={results['dP_crit_psi']:.0f} psi)",
    )

    if results["Pc_design_psi"] is not None:
        Pc = results["Pc_design_psi"]
        dP_design = inputs.OPERATING["P1_psi"] - Pc
        idx_design = results["idx_design"]
        primary_curve = results["nhne_curves"][kappa_primary]
        mdot_at_design = primary_curve[idx_design]

        ax.axvline(dP_design, color="#2ca02c", linestyle="-",
                   linewidth=1.5, alpha=0.6, zorder=1)
        ax.scatter(
            [dP_design], [mdot_at_design],
            s=90, marker="o", color="#2ca02c",
            edgecolor="#1f5e1f", linewidth=1.3, zorder=9,
            label=f"HRAP Pc={Pc:.0f} psi ({mdot_at_design:.3f} kg/s)",
        )

    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin / LB_TO_KG, ymax / LB_TO_KG)
    ax2.set_ylabel("ṁ (lb/s)", color="#555555")
    ax2.tick_params(axis="y", colors="#555555")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax.set_xlabel("Injector ΔP (psi)")
    ax.set_ylabel("mdot (kg/s)")

    inj = inputs.INJECTOR
    op = inputs.OPERATING
    upstream = results["upstream"]
    fig.suptitle(
        f"N2O Mass Flow vs ΔP — {inj['name']}",
        fontsize=14, fontweight="bold", y=0.97,
    )
    ax.set_title(
        f"Cd={inj['Cd']:.3f} · A={inj['A_physical']*1e6:.1f} mm² · "
        f"L/D={inj['L_over_D']:.2f} · "
        f"P₁={op['P1_psi']:.0f} psi · T₁={upstream['T1']:.1f} K",
        fontsize=11, color="#555555", pad=10,
    )

    legend = ax.legend(
        loc="upper left",
        framealpha=0.8,
        edgecolor="#cccccc",
        fontsize=10,
        ncol=1,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Plot:     {path}")


def save_metadata(results, kappa_primary, path):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "1.2",
        "weighting": "Solomon Eq. 2.21 (corrected from Dyer 2007)",
        "choke_definition": "Waxman Eq. 5: HEM peak (∂ṁ_HEM/∂P2 = 0)",
        "fluid": FLUID,
        "inputs": {
            "injector": inputs.INJECTOR,
            "operating": inputs.OPERATING,
            "kappa_sweep": inputs.KAPPA_SWEEP,
            "kappa_primary": kappa_primary,
            "sweep": inputs.SWEEP,
            "empirical_corrections": getattr(inputs, "EMPIRICAL_CORRECTIONS", None),
        },
        "upstream_state": {
            k: float(v) for k, v in results["upstream"].items()
        },
        "critical_point": {
            "P2_crit_psi": results["P2_crit_psi"],
            "dP_crit_psi": results["dP_crit_psi"],
            "P2_crit_over_Psat": results["P2_crit_over_Psat"],
            "mdot_crit_HEM_kgs": results["mdot_crit_HEM_kgs"],
        },
        "design_operating_point": (
            {
                "Pc_design_psi": results["Pc_design_psi"],
                "dP_design_psi": inputs.OPERATING["P1_psi"] - results["Pc_design_psi"],
                "is_choked": results["Pc_design_psi"] < results["P2_crit_psi"],
            }
            if results["Pc_design_psi"] is not None else None
        ),
        "operating_summary": {
            f"k_{k:.2f}": v for k, v in results["operating_summary"].items()
        },
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {path}")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print()
    print("Validating inputs...")
    validate_inputs()

    P1_pa = inputs.OPERATING["P1_psi"] * PSI_TO_PA
    Pc_design_psi = inputs.OPERATING.get("Pc_design_psi")

    results = run_model(
        Cd=inputs.INJECTOR["Cd"],
        A_physical=inputs.INJECTOR["A_physical"],
        P1=P1_pa,
        kappa_list=inputs.KAPPA_SWEEP,
        P2_min=inputs.SWEEP["P2_min_pa"],
        N_points=inputs.SWEEP["N_points"],
        Pc_design_psi=Pc_design_psi,
    )

    print_summary(results, inputs.KAPPA_PRIMARY)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Outputs")
    print("-" * 72)
    save_csv(results, os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["csv"]))
    save_plot(results, inputs.KAPPA_PRIMARY,
              os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["plot"]))
    save_metadata(results, inputs.KAPPA_PRIMARY,
                  os.path.join(OUTPUT_DIR, inputs.OUTPUT_FILES["metadata"]))
    print()
    print("Done.")
    print()