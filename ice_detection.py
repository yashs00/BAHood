"""
Ice Detection Module for LunarIce-360.

Implements multiple ice detection strategies using Chandrayaan-2 DFSAR
radar observables (CPR, DOP, polarimetric zone masks) and fuses them
into a single probability map.

Detection Methods
-----------------
1. Threshold-based  : CPR > thresh AND DOP < thresh
2. GMM clustering   : Gaussian Mixture Model on feature space
3. Anomaly detection: Isolation Forest for radar anomalies
4. Fusion           : Weighted combination with spatial cleanup
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_opening

from . import config


# =============================================================================
# THRESHOLD-BASED DETECTION
# =============================================================================

def threshold_detection(CPR_L, DOP_L, CPR_S=None,
                        cpr_thresh=None, dop_thresh=None):
    """
    Apply CPR/DOP thresholds to flag potential ice pixels.

    Ice candidates satisfy  CPR > cpr_thresh  AND  DOP < dop_thresh.
    When S-band CPR is available the detections are further split into
    *shallow ice* (flagged in **both** L-band and S-band) and *deep ice*
    (flagged in L-band only, consistent with ice below S-band penetration
    depth).

    Parameters
    ----------
    CPR_L : np.ndarray, shape (rows, cols)
        L-band Circular Polarisation Ratio.
    DOP_L : np.ndarray, shape (rows, cols)
        L-band Degree of Polarisation.
    CPR_S : np.ndarray or None, optional
        S-band CPR.  If provided, shallow / deep classification is
        performed.
    cpr_thresh : float, optional
        CPR threshold (default from ``config.CPR_ICE_THRESHOLD``).
    dop_thresh : float, optional
        DOP threshold (default from ``config.DOP_ICE_THRESHOLD``).

    Returns
    -------
    dict
        'ice_mask'    : bool array — combined ice candidate mask.
        'shallow_ice' : bool array or None — ice detected in both bands.
        'deep_ice'    : bool array or None — ice detected in L-band only.
    """
    if cpr_thresh is None:
        cpr_thresh = config.CPR_ICE_THRESHOLD
    if dop_thresh is None:
        dop_thresh = config.DOP_ICE_THRESHOLD

    ice_mask = (CPR_L > cpr_thresh) & (DOP_L < dop_thresh)

    shallow_ice = None
    deep_ice = None

    if CPR_S is not None:
        s_band_flag = CPR_S > cpr_thresh
        shallow_ice = ice_mask & s_band_flag   # both bands
        deep_ice = ice_mask & ~s_band_flag     # L-band only

    return {
        'ice_mask': ice_mask,
        'shallow_ice': shallow_ice,
        'deep_ice': deep_ice,
    }


# =============================================================================
# GMM-BASED DETECTION
# =============================================================================

def gmm_ice_detection(feature_stack, n_clusters=None):
    """
    Cluster radar features with a Gaussian Mixture Model and identify
    the cluster most consistent with subsurface ice.

    The ice cluster is selected as the one whose centre has the
    **highest CPR** (feature index 0) and **lowest DOP** (feature
    index 1).  A simple composite score
    ``centre_CPR - centre_DOP`` is used to rank clusters.

    Parameters
    ----------
    feature_stack : np.ndarray, shape (rows, cols, n_features)
        Multi-feature cube.  By convention the first two features
        are CPR and DOP respectively.
    n_clusters : int, optional
        Number of GMM components (default from ``config.GMM_N_CLUSTERS``).

    Returns
    -------
    dict
        'ice_probability' : np.ndarray (rows, cols) — posterior
            probability of belonging to the ice cluster.
        'labels'          : np.ndarray (rows, cols) — hard cluster
            assignments.
        'ice_cluster_id'  : int — index of the identified ice cluster.
        'model'           : fitted ``GaussianMixture`` object.
        'scaler'          : fitted ``StandardScaler`` object.
    """
    if n_clusters is None:
        n_clusters = config.GMM_N_CLUSTERS

    rows, cols, n_feat = feature_stack.shape

    # Flatten spatial dimensions → (n_pixels, n_features)
    X = feature_stack.reshape(-1, n_feat)

    # Filter out NaNs/Infs
    valid_mask = np.all(np.isfinite(X), axis=1)
    if not np.any(valid_mask):
        return {
            'ice_probability': np.zeros((rows, cols)),
            'labels': np.full((rows, cols), -1),
            'ice_cluster_id': -1,
            'model': None,
            'scaler': None,
        }

    X_valid = X[valid_mask]

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42,
                           covariance_type='full', max_iter=300)
    gmm.fit(X_scaled)

    # Cluster labels & posterior probabilities
    labels_valid = gmm.predict(X_scaled)
    probs_valid = gmm.predict_proba(X_scaled)          # (n_valid_pixels, n_clusters)

    # Identify ice cluster: un-scale centres back to physical units
    centres_physical = scaler.inverse_transform(gmm.means_)
    
    CPR_c = centres_physical[:, 0]
    DOP_c = centres_physical[:, 1]
    H_c = centres_physical[:, 5]
    alpha_c = centres_physical[:, 7]
    
    # Base score = CPR - DOP -> highest wins
    scores = CPR_c - DOP_c
    
    # Secondary validation: penalize clusters that are clearly not in 
    # H-Alpha ice zones (H > 0.5 and alpha > 42.5)
    penalty_mask = (H_c < 0.5) | (alpha_c < 40.0)
    scores[penalty_mask] -= 10.0
    
    ice_cluster_id = int(np.argmax(scores))

    # Reshape
    probs_flat = np.zeros((rows * cols, n_clusters))
    probs_flat[valid_mask] = probs_valid
    
    labels_flat = np.full(rows * cols, -1)
    labels_flat[valid_mask] = labels_valid

    ice_probability = probs_flat[:, ice_cluster_id].reshape(rows, cols)
    labels = labels_flat.reshape(rows, cols)

    return {
        'ice_probability': ice_probability,
        'labels': labels,
        'ice_cluster_id': ice_cluster_id,
        'model': gmm,
        'scaler': scaler,
    }


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def anomaly_detection(feature_stack, contamination=None):
    """
    Detect anomalous radar signatures using Isolation Forest.

    Anomaly scores are rescaled to [0, 1] where **1 = most anomalous**.

    Parameters
    ----------
    feature_stack : np.ndarray, shape (rows, cols, n_features)
        Multi-feature cube.
    contamination : float, optional
        Expected anomaly fraction (default from
        ``config.ISOLATION_FOREST_CONTAMINATION``).

    Returns
    -------
    np.ndarray, shape (rows, cols)
        Anomaly score map in [0, 1].
    """
    if contamination is None:
        contamination = config.ISOLATION_FOREST_CONTAMINATION

    rows, cols, n_feat = feature_stack.shape
    X = feature_stack.reshape(-1, n_feat)

    # Filter out NaNs/Infs
    valid_mask = np.all(np.isfinite(X), axis=1)
    if not np.any(valid_mask):
        return np.zeros((rows, cols))

    X_valid = X[valid_mask]

    # Normalise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Fit Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42,
                          n_estimators=200)
    iso.fit(X_scaled)

    # Raw scores: lower (more negative) = more anomalous
    raw_scores_valid = iso.decision_function(X_scaled)

    # Rescale to [0, 1], 1 = most anomalous
    score_min = raw_scores_valid.min()
    score_max = raw_scores_valid.max()
    denom = score_max - score_min
    
    anomaly_map = np.zeros(rows * cols)
    if denom > 0:
        anomaly_map[valid_mask] = 1.0 - (raw_scores_valid - score_min) / denom

    return anomaly_map.reshape(rows, cols)


# =============================================================================
# DETECTION FUSION
# =============================================================================

def fuse_detections(ice_prob_gmm, anomaly_map, threshold_mask,
                    zone_mask, weights=None):
    """
    Fuse multiple detection layers into a single ice probability map.

    Each input is treated as a [0, 1] evidence layer.  A weighted average
    produces the raw fused map, which is then cleaned with morphological
    opening to remove salt-and-pepper noise.

    Parameters
    ----------
    ice_prob_gmm : np.ndarray (rows, cols)
        GMM posterior probability for the ice cluster.
    anomaly_map : np.ndarray (rows, cols)
        Anomaly score in [0, 1].
    threshold_mask : np.ndarray (rows, cols)
        Binary threshold mask (True/1 = ice candidate).
    zone_mask : np.ndarray (rows, cols)
        H/alpha zone indicator (1 = ice-favourable zone, else 0).
    weights : dict or None, optional
        Keys 'gmm', 'anomaly', 'threshold', 'h_alpha_zone'.
        Defaults to ``config.FUSION_WEIGHTS``.

    Returns
    -------
    np.ndarray, shape (rows, cols)
        Fused ice probability map in [0, 1].
    """
    if weights is None:
        weights = config.FUSION_WEIGHTS

    w_gmm = weights.get('gmm', 0.35)
    w_anom = weights.get('anomaly', 0.25)
    w_thresh = weights.get('threshold', 0.20)
    w_zone = weights.get('h_alpha_zone', 0.20)

    # Cast boolean masks to float
    threshold_float = np.asarray(threshold_mask, dtype=np.float64)
    zone_float = np.asarray(zone_mask, dtype=np.float64)

    # Weighted average
    fused = (w_gmm * ice_prob_gmm
             + w_anom * anomaly_map
             + w_thresh * threshold_float
             + w_zone * zone_float)

    # Normalise by total weight
    total_w = w_gmm + w_anom + w_thresh + w_zone
    if total_w > 0:
        fused /= total_w

    # Spatial cleanup: binary opening on thresholded map, then mask
    binary_map = fused > 0.5
    struct = np.ones((3, 3), dtype=bool)
    cleaned = binary_opening(binary_map, structure=struct)

    # Zero-out pixels that didn't survive opening
    fused[~cleaned] = 0.0

    # Clip to [0, 1]
    np.clip(fused, 0.0, 1.0, out=fused)

    return fused


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_ice_detection(feature_stack, CPR_L, DOP_L, CPR_S=None,
                      zone_mask=None):
    """
    Run the full ice detection pipeline and return all intermediate results.

    Steps executed:
    1. Threshold detection (CPR / DOP).
    2. GMM clustering on the feature stack.
    3. Anomaly detection via Isolation Forest.
    4. Fusion of all evidence layers.

    Parameters
    ----------
    feature_stack : np.ndarray, shape (rows, cols, n_features)
        Multi-feature cube (CPR at index 0, DOP at index 1 by convention).
    CPR_L : np.ndarray (rows, cols)
        L-band CPR.
    DOP_L : np.ndarray (rows, cols)
        L-band DOP.
    CPR_S : np.ndarray or None, optional
        S-band CPR for dual-band analysis.
    zone_mask : np.ndarray or None, optional
        H/alpha classification zone mask (1 = ice-favourable).
        If ``None``, a zero-filled array is used (no zone information).

    Returns
    -------
    dict
        'threshold'        : dict from ``threshold_detection``.
        'gmm'              : dict from ``gmm_ice_detection``.
        'anomaly_map'      : np.ndarray from ``anomaly_detection``.
        'fused_probability' : np.ndarray from ``fuse_detections``.
    """
    rows, cols = CPR_L.shape

    # 1. Threshold detection
    thresh_result = threshold_detection(CPR_L, DOP_L, CPR_S=CPR_S)

    # 2. GMM clustering
    gmm_result = gmm_ice_detection(feature_stack)

    # 3. Anomaly detection
    anomaly_map = anomaly_detection(feature_stack)

    # 4. Zone mask fallback
    if zone_mask is None:
        zone_mask = np.zeros((rows, cols), dtype=np.float64)

    # 5. Fusion
    fused = fuse_detections(
        ice_prob_gmm=gmm_result['ice_probability'],
        anomaly_map=anomaly_map,
        threshold_mask=thresh_result['ice_mask'],
        zone_mask=zone_mask,
    )

    return {
        'threshold': thresh_result,
        'gmm': gmm_result,
        'anomaly_map': anomaly_map,
        'fused_probability': fused,
    }
