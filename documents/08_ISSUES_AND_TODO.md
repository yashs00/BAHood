# LunarIce-360 — Known Issues & Action Items

> Technical issues, inconsistencies, and improvements identified during codebase analysis. Address these before the hackathon.

---

## 🔴 Critical Issues

### 1. CPR Formula Inconsistency
**Location:** `polarimetry.py` vs `demo_synthetic.py` / `main.py` fallbacks

| File | Formula | Type |
|:---|:---|:---|
| `polarimetry.py` | `CPR = (S1 - S4) / (S1 + S4)` | Circular CPR ✅ |
| `demo_synthetic.py` fallback | `CPR = \|S1 - S2\| / \|S1 + S2\|` | Linear ratio ❌ |
| `main.py` fallback | `CPR = \|S1 - S2\| / \|S1 + S2\|` | Linear ratio ❌ |

> [!CAUTION]
> These compute entirely different quantities. The fallbacks will produce wrong results if the primary module fails.

**Fix:** Align all fallback CPR formulas to use `(S1 - S4) / (S1 + S4)`.

---

### 2. m-chi Decomposition Overwrites
**Location:** `polarimetry.py` lines ~136–150

The code redefines `Pv`, `Ps`, `Pd` multiple times. The final `Pv = S1*(1-m)` contradicts earlier computations from the m-chi ellipticity formulation.

**Fix:** Verify against Raney (2012) original paper. Ensure single, correct computation path.

---

### 3. Broken Test File
**Location:** `tests/test_ui_smoke.py`

```python
from ui_app import app  # Expects Flask app
```

Current `ui_app.py` uses Python stdlib `HTTPServer`, not Flask. This test will always fail.

**Fix:** Rewrite test to match current HTTP server implementation, or add proper tests.

---

## 🟡 Important Issues

### 4. MCMC Forward Model Calibration Constants
**Location:** `volume_estimation.py`

- `0.01` calibration factor in volume scattering model is arbitrary
- `0.2` baseline CPR in forward model has no physical justification
- These need validation against measured DFSAR data

**Fix:** Calibrate against known lunar terrain returns. Document assumptions clearly.

---

### 5. GMM Ice Cluster Selection Heuristic
**Location:** `ice_detection.py`

Current selection: pick cluster with `max(center_CPR - center_DOP)`

**Problem:** Rocky outcrops with high CPR could be selected as "ice cluster" if their CPR happens to be high relative to DOP.

**Fix:** Add secondary validation — check if selected cluster falls in H-Alpha ice zones (8, 9).

---

### 6. Fusion Weights Not Optimized
**Location:** `config.py`

Weights `(GMM=0.35, Anomaly=0.25, Threshold=0.20, H-alpha=0.20)` are hardcoded, not learned.

**Fix:** Add sensitivity analysis or cross-validation on synthetic data to justify weights.

---

### 7. Illumination Computation Performance
**Location:** `terrain.py`

Ray-tracing at 216 sun positions with per-step array shifting is O(n² × 216 × max_dist). Very slow for large DEMs.

**Fix:** Add option for downsampled computation with interpolation back to full resolution.

---

### 8. Traverse Connectivity Not Guaranteed
**Location:** `traverse.py`

NSGA-II optimizes waypoints independently. Straight-line interpolation between waypoints (20 points) may cross impassable terrain that interpolation misses.

**Fix:** Add collision checking along segments or use cost-aware interpolation (mini A* between waypoints).

---

## 🟢 Enhancements (Nice-to-Have)

### 9. No Georeferencing in Outputs
Outputs don't include lat/lon coordinates. Adding CRS transform would make outputs directly importable into GIS tools.

### 10. No Requirements.txt
Add a `requirements.txt` for easy dependency installation:
```
numpy
scipy
matplotlib
scikit-learn
emcee
corner
pymoo
rasterio
gdal
```

### 11. No Logging Framework
Uses `print()` statements. Switch to Python `logging` module for configurable verbosity.

### 12. Missing Sensitivity Analysis
No systematic analysis of how results change with threshold parameters (CPR threshold, DOP threshold, H-Alpha boundaries).

### 13. No ROC/AUC Analysis
For the ice detection fusion, compute ROC curves and AUC against synthetic ground truth to quantify detection performance.

---

## ✅ Pre-Hackathon Checklist

- [x] Fix CPR formula inconsistency (#1)
- [x] Verify m-chi decomposition (#2)
- [x] Fix or remove broken test (#3)
- [x] Add requirements.txt (#10)
- [ ] Run full pipeline on synthetic data — verify all 12 figures generate
- [ ] Test with varying grid sizes to ensure scalability
- [ ] Prepare for real DFSAR data format (GeoTIFF from PRADAN portal)
- [ ] Draft IEEE paper outline and abstract
- [ ] Prepare presentation template
