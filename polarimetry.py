"""
Polarimetry Module for LunarIce-360.

The central analysis module — computes all radar-polarimetric
observables needed for ice detection:

    * Circular Polarisation Ratio (CPR)
    * Degree of Polarisation (DOP)
    * m-chi decomposition (volume / surface / double-bounce powers)
    * Entropy (H), Anisotropy (A), Alpha angle — fast vectorised
    * Cloude–Pottier 9-zone classification
    * Dual-frequency differential features
    * Full 17-feature stack for machine learning

All functions use vectorised NumPy operations — no per-pixel loops.
"""

import numpy as np
from scipy.ndimage import uniform_filter

from . import config
from .preprocessing import to_db


# =============================================================================
# Basic Polarimetric Observables
# =============================================================================

def compute_CPR(S1, S4):
    """Compute the Circular Polarisation Ratio (CPR).

    .. math::

        \\text{CPR} = \\frac{S_1 - S_4}{S_1 + S_4}

    CPR > 1 indicates dominant volume / coherent backscatter
    (potential ice signature).

    Parameters
    ----------
    S1 : numpy.ndarray
        Total power Stokes parameter.
    S4 : numpy.ndarray
        Fourth Stokes parameter (circular polarisation difference).

    Returns
    -------
    CPR : numpy.ndarray
        Circular Polarisation Ratio array.  Pixels where
        ``S1 + S4 ≈ 0`` are set to ``np.nan``.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S4 = np.asarray(S4, dtype=np.float64)

    denom = S1 + S4
    CPR = np.where(np.abs(denom) > 1e-15,
                   (S1 - S4) / denom,
                   np.nan)
    return CPR


def compute_DOP(S1, S2, S3, S4):
    """Compute the Degree of Polarisation (DOP).

    .. math::

        \\text{DOP} = \\frac{\\sqrt{S_2^2 + S_3^2 + S_4^2}}{S_1}

    DOP is clipped to [0, 1].

    Parameters
    ----------
    S1, S2, S3, S4 : numpy.ndarray
        Stokes parameter arrays.

    Returns
    -------
    DOP : numpy.ndarray
        Degree of Polarisation, clipped to [0, 1].
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    S3 = np.asarray(S3, dtype=np.float64)
    S4 = np.asarray(S4, dtype=np.float64)

    polarised_power = np.sqrt(S2 ** 2 + S3 ** 2 + S4 ** 2)
    DOP = np.where(S1 > 1e-15, polarised_power / S1, 0.0)
    DOP = np.clip(DOP, 0.0, 1.0)
    return DOP


# =============================================================================
# m-chi Decomposition
# =============================================================================

def mchi_decomposition(S1, S2, S3, S4):
    """Perform the m-chi decomposition (Raney 2012).

    Decomposes the scattered field into volume, surface (even-bounce),
    and double-bounce (odd-bounce) scattering components using the
    Stokes vector.

    Parameters
    ----------
    S1, S2, S3, S4 : numpy.ndarray
        Stokes parameter arrays.

    Returns
    -------
    result : dict
        Keys:

        - ``'m'``   — Degree of polarisation (DOP).
        - ``'chi'`` — Ellipticity angle χ (radians).
        - ``'Pv'``  — Volume scattering power.
        - ``'Ps'``  — Surface (even-bounce) scattering power.
        - ``'Pd'``  — Double-bounce (odd-bounce) scattering power.

        All values are 2-D ``numpy.ndarray``.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    S3 = np.asarray(S3, dtype=np.float64)
    S4 = np.asarray(S4, dtype=np.float64)

    # Degree of polarisation
    m = compute_DOP(S1, S2, S3, S4)

    # Chi angle:  sin(2χ) = -S4 / (m * S1)
    m_S1 = m * S1
    sin2chi = np.where(m_S1 > 1e-15, -S4 / m_S1, 0.0)
    sin2chi = np.clip(sin2chi, -1.0, 1.0)
    chi = 0.5 * np.arcsin(sin2chi)

    # Scattering powers according to Raney (2012) m-chi decomposition
    Pv = S1 * (1.0 - m)                               # Volume (unpolarized)
    Ps = S1 * m * (1.0 + sin2chi) / 2.0               # Surface (even-bounce)
    Pd = S1 * m * (1.0 - sin2chi) / 2.0               # Double-bounce (odd-bounce)

    return {
        'm': m,
        'chi': chi,
        'Pv': Pv,
        'Ps': Ps,
        'Pd': Pd,
    }


# =============================================================================
# Fast Vectorised H / A / Alpha  (2×2 coherency matrix approach)
# =============================================================================

def compute_H_A_alpha(S1, S2, S3, S4, window_size=None):
    """Compute Entropy, Anisotropy, and Alpha angle — fast vectorised.

    Uses the reduced 2×2 coherency (Jones) matrix derived from
    Stokes parameters and an analytical eigenvalue solution
    (no per-pixel ``numpy.linalg.eig``).

    Algorithm
    ---------
    1. Spatially average Stokes parameters (``uniform_filter``).
    2. Build coherency matrix elements:

       .. math::

           J_{11} = (S_1 + S_2)/2, \\quad
           J_{22} = (S_1 - S_2)/2

           J_{12,\\text{re}} = S_3/2, \\quad
           J_{12,\\text{im}} = -S_4/2

    3. Analytical eigenvalues of the 2×2 Hermitian matrix:

       .. math::

           \\lambda_{1,2} = \\frac{\\text{tr} \\pm
           \\sqrt{\\text{tr}^2 - 4\\det}}{2}

    4. Derive H, A, α from eigenvalues and eigenvectors.

    Parameters
    ----------
    S1, S2, S3, S4 : numpy.ndarray
        Stokes parameter arrays (2-D).
    window_size : int, optional
        Spatial averaging window.  Defaults to
        ``config.SPATIAL_AVERAGING_WINDOW``.

    Returns
    -------
    result : dict
        Keys:

        - ``'H'``     — Polarimetric entropy ∈ [0, 1].
        - ``'A'``     — Anisotropy ∈ [0, 1].
        - ``'alpha'`` — Alpha angle in **degrees** ∈ [0, 90].
    """
    if window_size is None:
        window_size = config.SPATIAL_AVERAGING_WINDOW

    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    S3 = np.asarray(S3, dtype=np.float64)
    S4 = np.asarray(S4, dtype=np.float64)

    # --- Step 1: Spatial averaging ------------------------------------------
    s1 = uniform_filter(S1, size=window_size, mode='reflect')
    s2 = uniform_filter(S2, size=window_size, mode='reflect')
    s3 = uniform_filter(S3, size=window_size, mode='reflect')
    s4 = uniform_filter(S4, size=window_size, mode='reflect')

    # --- Step 2: Coherency matrix elements ----------------------------------
    J11 = (s1 + s2) / 2.0
    J22 = (s1 - s2) / 2.0
    J12_re = s3 / 2.0
    J12_im = -s4 / 2.0

    # --- Step 3: Analytical eigenvalues -------------------------------------
    trace = J11 + J22
    det = J11 * J22 - (J12_re ** 2 + J12_im ** 2)

    discriminant = np.maximum(trace ** 2 - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0

    # Ensure non-negative
    lambda1 = np.maximum(lambda1, 0.0)
    lambda2 = np.maximum(lambda2, 0.0)

    # --- Step 4: Entropy, Anisotropy, Alpha ---------------------------------

    # Probabilities (normalised eigenvalues)
    total = lambda1 + lambda2
    # Guard against zero total power
    safe_total = np.where(total > 1e-15, total, 1.0)

    p1 = lambda1 / safe_total
    p2 = lambda2 / safe_total

    # Where total ≈ 0, set probabilities to 0.5 each (max entropy)
    p1 = np.where(total > 1e-15, p1, 0.5)
    p2 = np.where(total > 1e-15, p2, 0.5)

    # Entropy: H = -Σ p_i log2(p_i)
    # Use safe log: log2(0) → 0 by convention
    log_p1 = np.where(p1 > 1e-15, np.log2(p1), 0.0)
    log_p2 = np.where(p2 > 1e-15, np.log2(p2), 0.0)
    H = -(p1 * log_p1 + p2 * log_p2)
    H = np.clip(H, 0.0, 1.0)

    # Anisotropy: A = (λ1 - λ2) / (λ1 + λ2)
    A = np.where(total > 1e-15,
                 (lambda1 - lambda2) / safe_total,
                 0.0)
    A = np.clip(A, 0.0, 1.0)

    # Alpha angle from eigenvectors of the 2×2 Hermitian matrix
    # For each pixel the dominant eigenvector of [[J11, J12],[J12*,J22]]
    # corresponding to λ1 is  v = [J12_re + j*J12_im,  λ1 - J11]
    #
    # α = arctan(|v2| / |v1|)  — the tilt from the surface-scatter axis

    v1_abs = np.sqrt(J12_re ** 2 + J12_im ** 2)
    v2_abs = np.abs(lambda1 - J11)

    # Guard: if both components are zero, default alpha = 45°
    denom_v = np.sqrt(v1_abs ** 2 + v2_abs ** 2)
    alpha_rad = np.where(
        denom_v > 1e-15,
        np.arctan2(v2_abs, v1_abs),
        np.pi / 4.0,
    )
    alpha_deg = np.degrees(alpha_rad)
    alpha_deg = np.clip(alpha_deg, 0.0, 90.0)

    return {
        'H': H,
        'A': A,
        'alpha': alpha_deg,
    }


# =============================================================================
# Cloude–Pottier 9-Zone Classification
# =============================================================================

def classify_H_alpha_zones(H, alpha,
                           h_low=None, h_high=None,
                           a_low=None, a_high=None):
    """Map H and α into the Cloude–Pottier 9-zone classification.

    The H–α plane is partitioned into 9 zones:

    +---------+-------------------+-------------------+-------------------+
    |         | Low α (<a_low)    | Med α             | High α (>a_high)  |
    +---------+-------------------+-------------------+-------------------+
    | High H  |    Zone 7         |    Zone 8         |    Zone 9         |
    | Med H   |    Zone 4         |    Zone 5         |    Zone 6         |
    | Low H   |    Zone 1         |    Zone 2         |    Zone 3         |
    +---------+-------------------+-------------------+-------------------+

    **Zones 8 and 9** (high entropy, medium–high α) are the primary
    ice-candidate regions.

    Parameters
    ----------
    H : numpy.ndarray
        Polarimetric entropy ∈ [0, 1].
    alpha : numpy.ndarray
        Alpha angle in degrees ∈ [0, 90].
    h_low : float, optional
        Low/medium entropy boundary.  Default ``config.H_LOW``.
    h_high : float, optional
        Medium/high entropy boundary.  Default ``config.H_HIGH``.
    a_low : float, optional
        Low/medium alpha boundary (degrees).  Default ``config.ALPHA_LOW``.
    a_high : float, optional
        Medium/high alpha boundary (degrees).  Default ``config.ALPHA_HIGH``.

    Returns
    -------
    zone_map : numpy.ndarray of int
        Integer zone labels 1–9.
    """
    if h_low is None:
        h_low = config.H_LOW
    if h_high is None:
        h_high = config.H_HIGH
    if a_low is None:
        a_low = config.ALPHA_LOW
    if a_high is None:
        a_high = config.ALPHA_HIGH

    H = np.asarray(H, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    zone_map = np.zeros_like(H, dtype=np.int32)

    # Entropy bands: 0 = low, 1 = medium, 2 = high
    h_band = np.zeros_like(H, dtype=np.int32)
    h_band[H >= h_low] = 1
    h_band[H >= h_high] = 2

    # Alpha bands: 0 = low, 1 = medium, 2 = high
    a_band = np.zeros_like(alpha, dtype=np.int32)
    a_band[alpha >= a_low] = 1
    a_band[alpha >= a_high] = 2

    # Zone = h_band * 3 + a_band + 1  (gives 1..9)
    zone_map = h_band * 3 + a_band + 1

    return zone_map


# =============================================================================
# Dual-Frequency Analysis
# =============================================================================

def dual_frequency_analysis(CPR_L, CPR_S, Pv_L, Pv_S):
    """Compute dual-frequency differential features.

    Parameters
    ----------
    CPR_L, CPR_S : numpy.ndarray
        CPR at L-band and S-band respectively.
    Pv_L, Pv_S : numpy.ndarray
        Volume scattering power at L-band and S-band.

    Returns
    -------
    result : dict
        Keys:

        - ``'CPR_diff'``   — ``CPR_L - CPR_S``.
        - ``'vol_ratio'``  — ``Pv_L / Pv_S`` (nan where Pv_S ≈ 0).
        - ``'CPR_ratio'``  — ``CPR_L / CPR_S`` (nan where CPR_S ≈ 0).
    """
    CPR_L = np.asarray(CPR_L, dtype=np.float64)
    CPR_S = np.asarray(CPR_S, dtype=np.float64)
    Pv_L = np.asarray(Pv_L, dtype=np.float64)
    Pv_S = np.asarray(Pv_S, dtype=np.float64)

    CPR_diff = CPR_L - CPR_S

    vol_ratio = np.where(np.abs(Pv_S) > 1e-15,
                         Pv_L / Pv_S,
                         np.nan)

    CPR_ratio = np.where(np.abs(CPR_S) > 1e-15,
                         CPR_L / CPR_S,
                         np.nan)

    return {
        'CPR_diff': CPR_diff,
        'vol_ratio': vol_ratio,
        'CPR_ratio': CPR_ratio,
    }


# =============================================================================
# Full Feature Stack Builder
# =============================================================================

def build_feature_stack(stokes_L, stokes_S):
    """Build the complete 17-feature stack for ML-based ice detection.

    Computes all polarimetric features from dual-frequency (L + S band)
    Stokes data and stacks them into a single 3-D array.

    Feature list (axis-2 order):

    ====  ============================  ==========
    Idx   Feature                       Band
    ====  ============================  ==========
     0    CPR                           L
     1    DOP                           L
     2    Pv  (volume)                  L
     3    Ps  (surface)                 L
     4    Pd  (double-bounce)           L
     5    H   (entropy)                 L
     6    A   (anisotropy)              L
     7    α   (alpha, degrees)          L
     8    σ⁰  (S1 in dB)               L
     9    CPR                           S
    10    DOP                           S
    11    Pv  (volume)                  S
    12    Ps  (surface)                 S
    13    Pd  (double-bounce)           S
    14    CPR_diff  (L − S)             dual
    15    vol_ratio (Pv_L / Pv_S)       dual
    16    CPR_ratio (CPR_L / CPR_S)     dual
    ====  ============================  ==========

    Parameters
    ----------
    stokes_L : dict
        L-band Stokes dict with keys ``'S1'``, ``'S2'``, ``'S3'``,
        ``'S4'`` — each a 2-D ``numpy.ndarray``.
    stokes_S : dict
        S-band Stokes dict (same structure).

    Returns
    -------
    feature_stack : numpy.ndarray
        3-D array of shape ``(rows, cols, 17)``.
    feature_names : list of str
        Ordered list of feature names.
    """
    # ---- L-band features ---------------------------------------------------
    S1_L, S2_L, S3_L, S4_L = (stokes_L['S1'], stokes_L['S2'],
                                stokes_L['S3'], stokes_L['S4'])

    CPR_L = compute_CPR(S1_L, S4_L)
    DOP_L = compute_DOP(S1_L, S2_L, S3_L, S4_L)
    mchi_L = mchi_decomposition(S1_L, S2_L, S3_L, S4_L)
    haa_L = compute_H_A_alpha(S1_L, S2_L, S3_L, S4_L)
    sigma0_L = to_db(S1_L)

    # ---- S-band features ---------------------------------------------------
    S1_S, S2_S, S3_S, S4_S = (stokes_S['S1'], stokes_S['S2'],
                                stokes_S['S3'], stokes_S['S4'])

    CPR_S = compute_CPR(S1_S, S4_S)
    DOP_S = compute_DOP(S1_S, S2_S, S3_S, S4_S)
    mchi_S = mchi_decomposition(S1_S, S2_S, S3_S, S4_S)

    # ---- Dual-frequency features -------------------------------------------
    df = dual_frequency_analysis(CPR_L, CPR_S, mchi_L['Pv'], mchi_S['Pv'])

    # ---- Stack all features ------------------------------------------------
    rows, cols = S1_L.shape

    feature_names = [
        'CPR_L', 'DOP_L',
        'Pv_L', 'Ps_L', 'Pd_L',
        'H_L', 'A_L', 'alpha_L',
        'sigma0_L',
        'CPR_S', 'DOP_S',
        'Pv_S', 'Ps_S', 'Pd_S',
        'CPR_diff', 'vol_ratio', 'CPR_ratio',
    ]

    feature_stack = np.empty((rows, cols, len(feature_names)),
                             dtype=np.float64)

    feature_stack[:, :, 0] = CPR_L
    feature_stack[:, :, 1] = DOP_L
    feature_stack[:, :, 2] = mchi_L['Pv']
    feature_stack[:, :, 3] = mchi_L['Ps']
    feature_stack[:, :, 4] = mchi_L['Pd']
    feature_stack[:, :, 5] = haa_L['H']
    feature_stack[:, :, 6] = haa_L['A']
    feature_stack[:, :, 7] = haa_L['alpha']
    feature_stack[:, :, 8] = sigma0_L
    feature_stack[:, :, 9] = CPR_S
    feature_stack[:, :, 10] = DOP_S
    feature_stack[:, :, 11] = mchi_S['Pv']
    feature_stack[:, :, 12] = mchi_S['Ps']
    feature_stack[:, :, 13] = mchi_S['Pd']
    feature_stack[:, :, 14] = df['CPR_diff']
    feature_stack[:, :, 15] = df['vol_ratio']
    feature_stack[:, :, 16] = df['CPR_ratio']

    print(f"[polarimetry] Built feature stack: "
          f"{feature_stack.shape}  ({len(feature_names)} features)")

    return feature_stack, feature_names
