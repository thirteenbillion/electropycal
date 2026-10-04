"""The feature catalog: one printable source of truth for every feature type the extractor emits.

The feature columns are hardcoded in the extractor (``features.extract``); this module gives them
human-readable definitions/derivations, units, role (leakage-safe **predictor** vs DA-derived
**RESERVED target**), and the D0-normalization kind, callable from anywhere::

    python -c "from electropycal import print_feature_catalog; print_feature_catalog()"

``feature_catalog()`` returns the same content as a list of dicts (or a DataFrame via ``.as_frame``)
so the notebooks and the CLI share one definition list instead of re-declaring it.
"""

from __future__ import annotations

# featuretype -> (definition / derivation, units, role). Per-frequency EIS types are emitted as
# <type>_fNN across the in-band grid; _integral / _band_* / _ratio are whole-spectrum summaries.
FEATURE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    # --- EIS per-frequency (Bode-derived), one column per in-band frequency ---
    "R_s": ("series resistance from the Bode fit (Re path)", "Ω", "predictor"),
    "R_p": ("parallel / charge-transfer resistance from the Bode fit", "Ω", "predictor"),
    "C_s": ("series capacitance = -1/(2·π·f·Im Z)", "F", "predictor"),
    "C_p": ("parallel capacitance (Im path)", "F", "predictor"),
    "ideality_C": ("phase ideality = sin²(phase); 1 = ideal capacitor, 0 = pure resistor", "", "predictor"),
    "tau": ("local RC time constant R·C", "s", "predictor"),
    "local_n": ("CPE exponent = -d log|Z| / d log f (1 = capacitive, 0 = resistive)", "", "predictor"),
    # --- EIS whole-spectrum summaries ---
    "R_s_integral": ("∫ R_s d(log f) over the band", "Ω·dec", "predictor"),
    "R_p_integral": ("∫ R_p d(log f) over the band", "Ω·dec", "predictor"),
    "C_s_integral": ("∫ C_s d(log f) over the band", "F·dec", "predictor"),
    "C_p_integral": ("∫ C_p d(log f) over the band", "F·dec", "predictor"),
    "f_ideality_crossover": ("frequency where ideality_C crosses 0.5 (capacitive→resistive)", "Hz", "predictor"),
    "ideality_C_band_LF/MF/HF": ("mean ideality_C in the low / mid / high sub-band", "", "predictor"),
    "n_band_LF/MF/HF": ("mean local_n (CPE exponent) in the low / mid / high sub-band", "", "predictor"),
    "tau_ratio": ("tau(low-f) / tau(high-f): dispersion of the time constant", "", "predictor"),
    "min_neg_phase": ("most-inductive phase across the FULL measured spectrum (severity; band-independent)", "°", "predictor"),
    "inductive_onset_hz": ("lowest frequency with Im Z ≥ 0 across the FULL spectrum (onset location; band-independent)", "Hz", "predictor"),
    # --- FSCV background-CV predictors (0 nM cycle; DA-independent) ---
    "mean_Vpeak": ("mean oxidation-peak potential V_ox of the bg-subtracted FSCV", "V", "predictor"),
    "mean_Ibg": ("mean background current at V_ox (0 nM), the NormIpeak denominator", "A", "predictor"),
    "bg_charge": ("anodic charging charge ∫ i_bg dV over the V_ox window (∝ C_dl × window)", "A·V", "predictor"),
    "bg_cap": ("double-layer-capacitance proxy: mean |i_anodic − i_cathodic|/2 (CV loop half-separation) over the window", "A", "predictor"),
    "bg_switch": ("background current at the anodic switching potential (solvent/electrode window edge; fouling)", "A", "predictor"),
    # --- targets (DA-derived; RESERVED: never predictors, to keep discovery leakage-free) ---
    "NormIpeak": ("(I_sig − I_bg)/I_bg at V_ox = bg-subtracted peak ÷ background, the response", "", "target"),
    "peak_height": ("bg-subtracted faradaic peak height (NormIpeak numerator)", "A", "target"),
    "peak_area": ("bg-subtracted peak area / charge", "A·V", "target"),
    "peak_fwhm": ("bg-subtracted peak full width at half maximum (kinetics)", "V", "target"),
    "sensitivity": ("per sensor-timepoint dose-response slope (NormIpeak vs log10[DA])", "", "target"),
    "sensitivity_intercept": ("dose-response intercept (DC level; signed → linear only)", "", "target"),
    "sensitivity_curvature": ("dose-response quadratic curvature (shape; signed → linear only)", "", "target"),
    "sat_imax": ("Langmuir saturation ceiling Imax (∝ active-site density) from NormIpeak = Imax·C/(Kd+C)", "", "target"),
    "sat_kd": ("Langmuir half-saturation concentration Kd", "nM", "target"),
    "sat_logkd": ("log10(Kd): Langmuir half-saturation (log scale)", "", "target"),
    "hill_imax": ("Hill saturation ceiling Imax from Imax·Cⁿ/(Kdⁿ+Cⁿ) (SNR-weighted fit)", "", "target"),
    "hill_kd": ("Hill half-saturation concentration Kd", "nM", "target"),
    "hill_n": ("Hill cooperativity / steepness exponent n (n=1 is Langmuir)", "", "target"),
    "power_beta": ("Freundlich power-law exponent β from NormIpeak = a·Cᵝ (β>1 supra-linear; non-saturating, fits straight/upward curves)", "", "target"),
    "power_a": ("Freundlich power-law coefficient a (response gain at unit concentration)", "", "target"),
    # --- QC / bookkeeping columns (RESERVED: describe the row; never predictors or targets) ---
    "noise_floor": ("non-faradaic FSCV noise floor below the peak (SNR-gate denominator)", "", "metadata"),
    "snr": ("NormIpeak / noise_floor (per-dose signal-to-noise)", "", "metadata"),
    "rms_snr": ("RMS signal-to-noise variant", "", "metadata"),
    "repeatability_snr": ("NormIpeak / replicate-to-replicate std across the first REPLICATES cycles", "", "metadata"),
    "rep_std": ("replicate-to-replicate std of NormIpeak", "", "metadata"),
    "dose_response_r": ("per channel-timepoint dose-response correlation (the FSCV.1 statistic)", "", "metadata"),
    "peak_at_edge": ("flag: located V_ox pinned to the peak-window edge (unreliable peak)", "", "metadata"),
    "peak_area_clipped": ("flag: faradaic lobe truncated by the sweep/window edge (peak_area is an "
                          "under-estimate; common at low dose, a voltage-sweep-span limit)", "", "metadata"),
    "n_conc": ("number of non-zero doses in the channel-timepoint's dose-response", "", "metadata"),
    # --- optional temporal predictors: catalogued for completeness, NOT emitted by
    #     extract_dataset; derive them per sensor from a featureset if wanted ---
    "time_since_baseline": ("days since the sensor's first timepoint (aging covariate; known at deployment)", "days", "predictor"),
    "<feature>__lag1": ("any predictor's value at the previous timepoint (per sensor, per dose)", "", "predictor"),
    "<feature>__delta": ("any predictor's first difference (change vs the previous timepoint; bounded, same scale)", "", "predictor"),
}


def _representative_column(featuretype: str) -> str:
    """A concrete column name for a type, so ``d0_normalization_kind`` can classify it."""
    if featuretype.endswith("_LF/MF/HF"):
        return featuretype[:-len("_LF/MF/HF")] + "_LF"
    return featuretype


def feature_catalog() -> list[dict]:
    """The feature dictionary as a list of dicts: ``featuretype, definition, units, role, d0_kind``."""
    from ..data.schema import d0_normalization_kind
    return [{"featuretype": t, "definition": defn, "units": units, "role": role,
             "d0_kind": d0_normalization_kind(_representative_column(t))}
            for t, (defn, units, role) in FEATURE_DEFINITIONS.items()]


def catalog_frame():
    """The catalog as a pandas DataFrame indexed by featuretype (requires pandas)."""
    import pandas as pd
    return pd.DataFrame(feature_catalog()).set_index("featuretype")


def print_feature_catalog() -> None:
    """Print the catalog as an aligned table (predictors first, then RESERVED targets)."""
    rows = feature_catalog()
    order = {"predictor": 0, "target": 1, "metadata": 2}
    rows.sort(key=lambda r: (order.get(r["role"], 3),))
    w_t = max(len(r["featuretype"]) for r in rows)
    w_u = max(3, max(len(r["units"]) for r in rows))
    print(f"{'featuretype':<{w_t}}  {'units':<{w_u}}  {'role':<9}  {'d0':<14}  definition")
    print("-" * (w_t + w_u + 9 + 14 + 20))
    for r in rows:
        print(f"{r['featuretype']:<{w_t}}  {r['units'] or '-':<{w_u}}  {r['role']:<9}  "
              f"{r['d0_kind']:<14}  {r['definition']}")
    from collections import Counter
    c = Counter(r["role"] for r in rows)
    print(f"\n{c['predictor']} predictor types + {c['target']} RESERVED target types + "
          f"{c['metadata']} QC/metadata columns. Predictors feed the model; DA-derived targets are "
          "never predictors (leakage-free); metadata describe the row.")
