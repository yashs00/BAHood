"""
Synthetic Data Generator & Demo for LunarIce-360.

Generates synthetic DFSAR-like data for testing the full pipeline without
requiring actual Chandrayaan-2 data. The synthetic data includes a DEM
with craters, Stokes parameters with injected ice signatures, and ground
truth masks.

Usage:
    python -m lunarice360.demo_synthetic
"""

import numpy as np
import os
import sys

from . import config


# =============================================================================
# SYNTHETIC DEM GENERATION
# =============================================================================

def generate_synthetic_dem(rows=500, cols=500, pixel_size=20.0):
    """Generate a synthetic Digital Elevation Model with craters.

    Creates a DEM featuring:
    - A large flat plain as base terrain
    - A large crater (circular depression with raised rim)
    - A smaller 'doubly shadowed' crater inside the large one

    Parameters
    ----------
    rows : int
        Number of rows in the DEM.
    cols : int
        Number of columns in the DEM.
    pixel_size : float
        Pixel size in meters.

    Returns
    -------
    dem : np.ndarray
        Synthetic DEM of shape (rows, cols) in meters.
    pixel_size : float
        The pixel size used.
    """
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # Base terrain: gently sloping plain with subtle large-scale undulation
    dem = (
        -2000.0  # Base elevation (meters, relative)
        + 0.5 * y  # Gentle north-south slope
        + 30.0 * np.sin(2 * np.pi * x / cols) *
          np.cos(2 * np.pi * y / rows)  # Undulation
    )

    # --- Large Crater ---
    crater_center = (rows // 2, cols // 2)  # Center of image
    crater_radius_px = min(rows, cols) // 4  # ~125 pixels
    crater_depth = 400.0  # meters
    rim_height = 80.0  # meters

    dist_crater = np.sqrt(
        (y - crater_center[0])**2 + (x - crater_center[1])**2
    )
    # Normalized distance from crater center
    r_norm = dist_crater / crater_radius_px

    # Crater interior: parabolic depression
    crater_mask = r_norm <= 1.0
    dem[crater_mask] -= crater_depth * (1.0 - r_norm[crater_mask]**2)

    # Raised rim: Gaussian ring just outside crater edge
    rim_mask = (r_norm > 0.9) & (r_norm < 1.3)
    rim_profile = rim_height * np.exp(-((r_norm - 1.0) / 0.1)**2)
    dem[rim_mask] += rim_profile[rim_mask]

    # --- Small Inner Crater (Doubly Shadowed Region) ---
    inner_center = (crater_center[0] + 30, crater_center[1] + 20)
    inner_radius_px = crater_radius_px // 4  # ~31 pixels
    inner_depth = 150.0

    dist_inner = np.sqrt(
        (y - inner_center[0])**2 + (x - inner_center[1])**2
    )
    r_inner_norm = dist_inner / inner_radius_px

    inner_mask = r_inner_norm <= 1.0
    dem[inner_mask] -= inner_depth * (1.0 - r_inner_norm[inner_mask]**2)

    # Small inner rim
    inner_rim_mask = (r_inner_norm > 0.85) & (r_inner_norm < 1.2)
    inner_rim_profile = 30.0 * np.exp(-((r_inner_norm - 1.0) / 0.08)**2)
    dem[inner_rim_mask] += inner_rim_profile[inner_rim_mask]

    # Add small-scale noise (surface roughness)
    dem += np.random.normal(0, 2.0, dem.shape)

    return dem, pixel_size


# =============================================================================
# SYNTHETIC STOKES PARAMETER GENERATION
# =============================================================================

def generate_synthetic_stokes(dem, band='L', ice_center=(300, 300),
                               ice_radius=40):
    """Generate synthetic Stokes parameters mimicking DFSAR data.

    Creates Stokes vector [S1, S2, S3, S4] with:
    - Background: normal surface scattering (low CPR, moderate DOP)
    - Ice region: high CPR (>1), low DOP (<0.13), high volume scattering

    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model (used for terrain-correlated backscatter).
    band : str
        'L' for L-band or 'S' for S-band.
    ice_center : tuple
        (row, col) center of the ice deposit.
    ice_radius : int
        Radius of the ice region in pixels.

    Returns
    -------
    dict
        Dictionary with keys 'S1', 'S2', 'S3', 'S4', each a 2D array.
    """
    rows, cols = dem.shape
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # --- Background (Normal Surface) ---
    # S1: total backscatter power (terrain-correlated)
    slope_proxy = np.sqrt(
        np.gradient(dem, axis=0)**2 + np.gradient(dem, axis=1)**2
    )
    S1_base = 0.8 + 0.3 * np.tanh(slope_proxy / 50.0)
    S1 = S1_base + np.random.exponential(0.05, (rows, cols))

    # For normal surface: CPR ~ 0.3, DOP ~ 0.5-0.8
    # CPR = (S1 - S4) / (S1 + S4) => S4 = S1 * (1 - CPR) / (1 + CPR)
    background_cpr = 0.3 + np.random.normal(0, 0.05, (rows, cols))
    background_cpr = np.clip(background_cpr, 0.05, 0.7)
    S4 = S1 * (1 - background_cpr) / (1 + background_cpr)

    # S2, S3: remaining polarized components
    # DOP = sqrt(S2^2 + S3^2 + S4^2) / S1
    # For moderate DOP (~0.6): S2 and S3 are moderate
    background_dop = 0.6 + np.random.normal(0, 0.1, (rows, cols))
    background_dop = np.clip(background_dop, 0.3, 0.9)
    remaining_power = np.sqrt(
        np.maximum(0, (background_dop * S1)**2 - S4**2)
    )
    phase = np.random.uniform(0, 2 * np.pi, (rows, cols))
    S2 = remaining_power * np.cos(phase) * 0.7
    S3 = remaining_power * np.sin(phase) * 0.7

    # Add speckle noise (multiplicative)
    speckle = np.random.gamma(5, 0.2, (rows, cols))
    S1 *= speckle
    S2 *= speckle * np.random.gamma(5, 0.2, (rows, cols))

    # --- Ice Region ---
    dist_from_ice = np.sqrt(
        (y - ice_center[0])**2 + (x - ice_center[1])**2
    )
    # Smooth transition at ice boundary
    ice_weight = np.clip(1.0 - (dist_from_ice / ice_radius), 0, 1)
    ice_weight = ice_weight**2  # Sharper boundary

    # Band-dependent ice signature strength
    if band.upper() == 'L':
        # L-band: deeper penetration, stronger ice signatures
        ice_cpr = 1.4 + np.random.normal(0, 0.15, (rows, cols))
        ice_dop = 0.08 + np.random.normal(0, 0.02, (rows, cols))
        ice_s1_boost = 1.5
    else:
        # S-band: shallower penetration, weaker signatures
        ice_cpr = 1.1 + np.random.normal(0, 0.2, (rows, cols))
        ice_dop = 0.11 + np.random.normal(0, 0.03, (rows, cols))
        ice_s1_boost = 1.2

    ice_cpr = np.clip(ice_cpr, 0.8, 2.5)
    ice_dop = np.clip(ice_dop, 0.01, 0.25)

    # Apply ice signatures with smooth blending
    S1_ice = S1 * ice_s1_boost
    S4_ice = S1_ice * (1 - ice_cpr) / (1 + ice_cpr)
    remaining_ice = np.sqrt(
        np.maximum(0, (ice_dop * S1_ice)**2 - S4_ice**2)
    )
    S2_ice = remaining_ice * np.cos(phase) * 0.5
    S3_ice = remaining_ice * np.sin(phase) * 0.5

    # Blend ice and background
    S1 = S1 * (1 - ice_weight) + S1_ice * ice_weight
    S2 = S2 * (1 - ice_weight) + S2_ice * ice_weight
    S3 = S3 * (1 - ice_weight) + S3_ice * ice_weight
    S4 = S4 * (1 - ice_weight) + S4_ice * ice_weight

    # Ensure S1 is positive
    S1 = np.maximum(S1, 1e-6)

    return {'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4}


# =============================================================================
# COMPLETE SYNTHETIC DATASET
# =============================================================================

def generate_synthetic_dataset(rows=500, cols=500):
    """Generate a complete synthetic dataset for pipeline testing.

    Creates DEM with craters and dual-band (L + S) Stokes parameters
    with injected ice signatures inside the deep crater.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dem': synthetic DEM (np.ndarray)
        - 'pixel_size': pixel size in meters (float)
        - 'stokes_L': L-band Stokes dict {S1, S2, S3, S4}
        - 'stokes_S': S-band Stokes dict {S1, S2, S3, S4}
        - 'ice_truth_mask': ground truth binary mask (np.ndarray)
        - 'crater_center': (row, col) of the main crater center
    """
    print("="*60)
    print("  LunarIce-360: Generating Synthetic Dataset")
    print("="*60)

    # Generate DEM
    print("\n[1/4] Generating synthetic DEM...")
    dem, pixel_size = generate_synthetic_dem(rows, cols)
    crater_center = (rows // 2, cols // 2)
    print(f"  DEM shape: {dem.shape}")
    print(f"  Elevation range: [{dem.min():.1f}, {dem.max():.1f}] m")
    print(f"  Pixel size: {pixel_size} m")

    # Ice deposit location: inside the inner crater
    ice_center = (crater_center[0] + 30, crater_center[1] + 20)
    ice_radius = 35

    # Generate ground truth mask
    print("\n[2/4] Creating ground truth ice mask...")
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dist_ice = np.sqrt(
        (y - ice_center[0])**2 + (x - ice_center[1])**2
    )
    ice_truth_mask = (dist_ice <= ice_radius).astype(np.float64)
    n_ice_pixels = int(ice_truth_mask.sum())
    print(f"  Ice region center: {ice_center}")
    print(f"  Ice region radius: {ice_radius} pixels")
    print(f"  Ice pixels: {n_ice_pixels} ({100*n_ice_pixels/(rows*cols):.2f}%)")

    # Generate L-band Stokes
    print("\n[3/4] Generating L-band Stokes parameters...")
    stokes_L = generate_synthetic_stokes(dem, band='L',
                                         ice_center=ice_center,
                                         ice_radius=ice_radius)
    print(f"  S1 range: [{stokes_L['S1'].min():.4f}, "
          f"{stokes_L['S1'].max():.4f}]")

    # Generate S-band Stokes (weaker ice signatures)
    print("\n[4/4] Generating S-band Stokes parameters...")
    stokes_S = generate_synthetic_stokes(dem, band='S',
                                         ice_center=ice_center,
                                         ice_radius=ice_radius)
    print(f"  S1 range: [{stokes_S['S1'].min():.4f}, "
          f"{stokes_S['S1'].max():.4f}]")

    print("\n" + "="*60)
    print("  Synthetic dataset generation complete!")
    print("="*60)

    return {
        'dem': dem,
        'pixel_size': pixel_size,
        'stokes_L': stokes_L,
        'stokes_S': stokes_S,
        'ice_truth_mask': ice_truth_mask,
        'crater_center': crater_center,
    }


# =============================================================================
# DEMO RUNNER
# =============================================================================

def run_demo():
    """Run the full LunarIce-360 demo with synthetic data.

    Generates synthetic data, runs the complete pipeline (or replicates
    key steps inline if pipeline modules are not available), prints
    results, and saves all figures.

    This function is designed to be runnable as:
        python -m lunarice360.demo_synthetic
    """
    print()
    print("#" * 64)
    print("#" + " LunarIce-360: FULL DEMO WITH SYNTHETIC DATA ".center(62) + "#")
    print("#" * 64)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic dataset
    # -------------------------------------------------------------------------
    dataset = generate_synthetic_dataset(rows=500, cols=500)
    dem = dataset['dem']
    pixel_size = dataset['pixel_size']
    stokes_L = dataset['stokes_L']
    stokes_S = dataset['stokes_S']
    ice_truth = dataset['ice_truth_mask']
    crater_center = dataset['crater_center']

    # -------------------------------------------------------------------------
    # Try to run the full pipeline via main.py
    # -------------------------------------------------------------------------
    try:
        from .main import run_full_pipeline
        print("\n>>> Running full pipeline via main.run_full_pipeline...")
        results = run_full_pipeline(
            stokes_L=stokes_L,
            stokes_S=stokes_S,
            dem=dem,
            pixel_size=pixel_size,
            target_center=crater_center,
            output_dir=config.OUTPUT_DIR,
        )
        print("\n>>> Full pipeline completed successfully!")

    except (ImportError, Exception) as e:
        print(f"\n>>> Full pipeline not available ({type(e).__name__}: {e})")
        print(">>> Running standalone demo steps...\n")

        # -----------------------------------------------------------------
        # Standalone demo: replicate key pipeline steps inline
        # -----------------------------------------------------------------
        from .visualization import (
            setup_output_dir,
            plot_ice_probability_map,
            plot_H_alpha_plane,
            plot_decomposition_panels,
            plot_terrain_analysis,
            plot_landing_site,
            plot_traverse_path,
            plot_pareto_front,
            plot_energy_profile,
            plot_volume_summary,
            plot_dual_frequency,
            generate_summary_figure,
        )

        setup_output_dir()
        results = {}

        # --- Polarimetric Feature Extraction (inline) ---
        print("[DEMO] Computing polarimetric features...")
        S1, S2, S3, S4 = (
            stokes_L['S1'], stokes_L['S2'],
            stokes_L['S3'], stokes_L['S4']
        )

        # CPR = (S1 - S2) / (S1 + S2)
        CPR_L = np.abs((S1 - S2) / (S1 + S2 + 1e-10))
        # DOP = sqrt(S2^2 + S3^2 + S4^2) / S1
        DOP_L = np.sqrt(S2**2 + S3**2 + S4**2) / (S1 + 1e-10)

        S1s, S2s, S3s, S4s = (
            stokes_S['S1'], stokes_S['S2'],
            stokes_S['S3'], stokes_S['S4']
        )
        CPR_S = np.abs((S1s - S2s) / (S1s + S2s + 1e-10))

        print(f"  CPR_L range: [{CPR_L.min():.3f}, {CPR_L.max():.3f}]")
        print(f"  DOP_L range: [{DOP_L.min():.3f}, {DOP_L.max():.3f}]")

        # --- m-chi decomposition (inline) ---
        print("[DEMO] Computing m-chi decomposition...")
        m = DOP_L
        chi = 0.5 * np.arctan2(S4, np.sqrt(S2**2 + S3**2 + 1e-10))
        Pv = S1 * m * (1 - np.sin(2 * chi)) / 2.0
        Ps = S1 * m * (1 + np.sin(2 * chi)) / 2.0
        Pd = S1 * (1 - m)
        # Normalize
        total_P = Pv + Ps + Pd + 1e-10
        Pv /= total_P
        Ps /= total_P
        Pd /= total_P

        # --- Synthetic H/alpha ---
        print("[DEMO] Computing synthetic H/alpha parameters...")
        H = np.clip(0.3 + 0.4 * Pv + np.random.normal(0, 0.05, dem.shape),
                    0, 1)
        alpha = np.clip(
            30 + 30 * Pv + np.random.normal(0, 3, dem.shape), 0, 90
        )

        # --- Ice Detection (threshold-based, inline) ---
        print("[DEMO] Running ice detection (threshold)...")
        ice_cpr_mask = CPR_L > config.CPR_ICE_THRESHOLD
        ice_dop_mask = DOP_L < config.DOP_ICE_THRESHOLD
        ice_detected = (ice_cpr_mask & ice_dop_mask).astype(np.float64)

        # Create probability map (combine with smooth weighting)
        ice_prob = np.zeros_like(CPR_L)
        ice_prob += 0.4 * np.clip((CPR_L - 0.8) / 0.8, 0, 1)
        ice_prob += 0.3 * np.clip((0.2 - DOP_L) / 0.2, 0, 1)
        ice_prob += 0.3 * np.clip(Pv / (Pv.max() + 1e-10), 0, 1)
        ice_prob = np.clip(ice_prob, 0, 1)

        results['ice_prob'] = ice_prob
        results['ice_detected'] = ice_detected
        results['CPR_L'] = CPR_L
        results['CPR_S'] = CPR_S

        n_detected = int(ice_detected.sum())
        print(f"  Detected ice pixels: {n_detected}")
        print(f"  Ice probability max: {ice_prob.max():.3f}")

        # --- Terrain Analysis (inline) ---
        print("[DEMO] Computing terrain analysis...")
        dy, dx = np.gradient(dem, pixel_size)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        from scipy.ndimage import uniform_filter
        roughness = np.sqrt(
            uniform_filter(dem**2, size=config.ROUGHNESS_WINDOW) -
            uniform_filter(dem, size=config.ROUGHNESS_WINDOW)**2
        )

        # Synthetic illumination (based on elevation relative to surroundings)
        dem_smooth = uniform_filter(dem, size=21)
        illumination = np.clip(
            0.5 + 0.5 * (dem - dem_smooth) / (np.std(dem) + 1e-10),
            0, 1
        )

        # Hazard map
        slope_norm = np.clip(slope / config.ROVER_MAX_SLOPE, 0, 1)
        rough_norm = np.clip(roughness / (roughness.max() + 1e-10), 0, 1)
        hazard = 0.5 * slope_norm + 0.3 * rough_norm + 0.2 * (1 - illumination)
        hazard = np.clip(hazard, 0, 1)

        results['slope'] = slope
        results['roughness'] = roughness
        results['illumination'] = illumination
        results['hazard'] = hazard

        # --- Landing Site Selection (inline) ---
        print("[DEMO] Selecting landing sites...")
        safety = 1.0 - hazard
        ice_center = (crater_center[0] + 30, crater_center[1] + 20)
        rows, cols = dem.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        dist_to_ice = np.sqrt(
            (y - ice_center[0])**2 + (x - ice_center[1])**2
        ) * pixel_size
        proximity = np.exp(
            -((dist_to_ice - config.LANDING_IDEAL_DISTANCE)**2) /
            (2 * config.LANDING_DISTANCE_SIGMA**2)
        )

        landing_score = (
            config.LANDING_WEIGHTS['safety'] * safety +
            config.LANDING_WEIGHTS['illumination'] * illumination +
            config.LANDING_WEIGHTS['proximity'] * proximity +
            config.LANDING_WEIGHTS['flatness'] * (1.0 - slope_norm)
        )

        # Mask unsafe areas
        landing_score[slope > config.LANDING_MAX_SLOPE] = 0
        landing_score[illumination < config.ILLUMINATION_MIN_FRACTION] = 0

        # Find top 3 sites (peak finding)
        from scipy.ndimage import maximum_filter
        local_max = (landing_score == maximum_filter(landing_score, size=30))
        local_max &= landing_score > 0.3
        site_coords = np.argwhere(local_max)
        if len(site_coords) > 0:
            site_scores = landing_score[site_coords[:, 0], site_coords[:, 1]]
            top_idx = np.argsort(site_scores)[::-1][:3]
            best_sites = []
            for rank, idx in enumerate(top_idx):
                r, c = site_coords[idx]
                best_sites.append({
                    'row': int(r), 'col': int(c),
                    'score': float(site_scores[idx]),
                    'label': f'Site {rank+1}'
                })
        else:
            # Fallback: use the pixel with highest score
            best_idx = np.unravel_index(
                np.argmax(landing_score), landing_score.shape
            )
            best_sites = [{
                'row': int(best_idx[0]), 'col': int(best_idx[1]),
                'score': float(landing_score[best_idx]),
                'label': 'Site 1'
            }]

        results['landing_scores'] = landing_score
        results['best_sites'] = best_sites
        for s in best_sites:
            print(f"  {s['label']}: row={s['row']}, col={s['col']}, "
                  f"score={s['score']:.3f}")

        # --- Rover Traverse Path (simplified A* inline) ---
        print("[DEMO] Planning rover traverse (simplified)...")
        start = (best_sites[0]['row'], best_sites[0]['col'])
        goal = ice_center

        # Simple straight-line path with waypoints
        n_waypoints = config.NSGA2_N_WAYPOINTS + 2
        path_rows = np.linspace(start[0], goal[0], n_waypoints).astype(int)
        path_cols = np.linspace(start[1], goal[1], n_waypoints).astype(int)
        path = np.column_stack([path_rows, path_cols])

        # Compute path metrics
        diffs = np.diff(path, axis=0) * pixel_size
        segment_dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        total_distance = segment_dists.sum()
        travel_time_hours = total_distance / config.ROVER_SPEED / 3600.0

        results['path'] = path
        results['total_distance'] = total_distance
        results['travel_time_hours'] = travel_time_hours
        print(f"  Path waypoints: {len(path)}")
        print(f"  Total distance: {total_distance:.1f} m")
        print(f"  Travel time: {travel_time_hours:.1f} hours")

        # --- Synthetic Pareto Front ---
        print("[DEMO] Generating synthetic Pareto front...")
        n_pareto = 50
        pareto_F = np.column_stack([
            np.random.uniform(2000, 5000, n_pareto),
            np.random.uniform(0.1, 0.8, n_pareto),
            np.random.uniform(0.2, 0.6, n_pareto),
        ])
        # Make it look Pareto-ish
        pareto_F = pareto_F[
            np.argsort(pareto_F[:, 0] + pareto_F[:, 1] * 5000 +
                       pareto_F[:, 2] * 3000)
        ]
        results['pareto_F'] = pareto_F

        # --- Energy Profile ---
        print("[DEMO] Computing energy profile...")
        cum_dist = np.concatenate([[0], np.cumsum(segment_dists)])
        n_steps = len(cum_dist)
        illuminated = np.array([
            illumination[int(r), int(c)] > 0.5
            for r, c in path
        ])
        battery = np.zeros(n_steps)
        battery[0] = config.ROVER_BATTERY_CAPACITY
        for i in range(1, n_steps):
            dt = segment_dists[i-1] / config.ROVER_SPEED  # seconds
            power_in = config.ROVER_SOLAR_POWER if illuminated[i] else 0
            power_out = config.ROVER_LOCOMOTION_POWER
            energy_delta = (power_in - power_out) * dt / 3600.0  # Wh
            battery[i] = np.clip(
                battery[i-1] + energy_delta,
                0, config.ROVER_BATTERY_CAPACITY
            )

        energy_profile = {
            'cumulative_distance': cum_dist,
            'battery_energy': battery,
            'illuminated': illuminated,
            'battery_max': config.ROVER_BATTERY_CAPACITY,
        }
        results['energy_profile'] = energy_profile

        # --- Ice Volume Estimation (simplified MCMC inline) ---
        print("[DEMO] Running simplified ice volume estimation...")
        ice_area_m2 = float(ice_detected.sum()) * pixel_size**2
        # Simplified: sample ice fraction, depth, compute volume
        n_samples = 2000
        frac_samples = np.random.beta(2, 20, n_samples)
        depth_samples = np.random.uniform(
            config.PRIOR_DEPTH[0], config.PRIOR_DEPTH[1], n_samples
        )
        roughness_samples = np.random.uniform(
            config.PRIOR_ROUGHNESS_CM[0], config.PRIOR_ROUGHNESS_CM[1],
            n_samples
        )
        density_samples = np.random.normal(
            config.REGOLITH_DENSITY_TYPICAL, 100, n_samples
        )

        volume_samples = ice_area_m2 * frac_samples * depth_samples

        mcmc_samples = np.column_stack([
            frac_samples, roughness_samples,
            density_samples, depth_samples
        ])

        volume_results = {
            'volume_samples': volume_samples,
            'median': float(np.median(volume_samples)),
            'ci_68_low': float(np.percentile(volume_samples, 16)),
            'ci_68_high': float(np.percentile(volume_samples, 84)),
            'ci_95_low': float(np.percentile(volume_samples, 2.5)),
            'ci_95_high': float(np.percentile(volume_samples, 97.5)),
        }
        results['volume_results'] = volume_results
        results['mcmc_samples'] = mcmc_samples
        print(f"  Ice area: {ice_area_m2:.0f} m²")
        print(f"  Volume estimate: {volume_results['median']:.1f} "
              f"[{volume_results['ci_68_low']:.1f}, "
              f"{volume_results['ci_68_high']:.1f}] m³ (68% CI)")

        # -----------------------------------------------------------------
        # Generate All Visualizations
        # -----------------------------------------------------------------
        print("\n" + "="*60)
        print("  Generating Visualizations")
        print("="*60)

        print("\n[VIZ 1/11] Ice probability map...")
        plot_ice_probability_map(ice_prob, dem=dem)

        print("[VIZ 2/11] H/alpha plane...")
        plot_H_alpha_plane(H, alpha)

        print("[VIZ 3/11] Decomposition panels...")
        plot_decomposition_panels(Pv, Ps, Pd)

        print("[VIZ 4/11] Terrain analysis...")
        plot_terrain_analysis(slope, roughness, illumination, hazard)

        print("[VIZ 5/11] Landing site selection...")
        plot_landing_site(landing_score, best_sites, dem=dem)

        print("[VIZ 6/11] Rover traverse path...")
        plot_traverse_path(path, dem, hazard_map=hazard,
                          ice_prob=ice_prob,
                          start=start, goal=goal)

        print("[VIZ 7/11] Pareto front...")
        plot_pareto_front(pareto_F, selected_idx=0)

        print("[VIZ 8/11] Energy profile...")
        plot_energy_profile(energy_profile)

        print("[VIZ 9/11] Volume summary...")
        plot_volume_summary(volume_results)

        print("[VIZ 10/11] Dual-frequency analysis...")
        plot_dual_frequency(CPR_L, CPR_S)

        print("[VIZ 11/11] Summary figure...")
        generate_summary_figure(ice_prob, hazard, path, volume_results)

        # Try MCMC corner plot (may need corner package)
        try:
            from .visualization import plot_mcmc_corner
            print("[VIZ BONUS] MCMC corner plot...")
            plot_mcmc_corner(mcmc_samples)
        except Exception as corner_err:
            print(f"  Skipping corner plot: {corner_err}")

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    print("\n")
    print("#" * 64)
    print("#" + " DEMO RESULTS SUMMARY ".center(62) + "#")
    print("#" * 64)

    if 'ice_prob' in results:
        ice_p = results['ice_prob']
        print(f"\n  Ice Detection:")
        print(f"    Max probability:    {ice_p.max():.3f}")
        print(f"    Pixels > 0.5 prob:  {int((ice_p > 0.5).sum())}")
        print(f"    Pixels > 0.8 prob:  {int((ice_p > 0.8).sum())}")

    if 'best_sites' in results:
        print(f"\n  Landing Sites:")
        for s in results['best_sites']:
            print(f"    {s['label']}: ({s['row']}, {s['col']}) "
                  f"score={s['score']:.3f}")

    if 'total_distance' in results:
        print(f"\n  Traverse:")
        print(f"    Distance:  {results['total_distance']:.1f} m")
        print(f"    Time:      {results['travel_time_hours']:.1f} hours")

    if 'volume_results' in results:
        vr = results['volume_results']
        print(f"\n  Ice Volume:")
        print(f"    Median:    {vr['median']:.1f} m³")
        print(f"    68% CI:    [{vr['ci_68_low']:.1f}, "
              f"{vr['ci_68_high']:.1f}] m³")
        print(f"    95% CI:    [{vr['ci_95_low']:.1f}, "
              f"{vr['ci_95_high']:.1f}] m³")

    # Accuracy vs ground truth
    if 'ice_detected' in results:
        det = results['ice_detected']
        tp = float(((det > 0) & (ice_truth > 0)).sum())
        fp = float(((det > 0) & (ice_truth == 0)).sum())
        fn = float(((det == 0) & (ice_truth > 0)).sum())
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        print(f"\n  Detection Accuracy (vs ground truth):")
        print(f"    Precision:  {precision:.3f}")
        print(f"    Recall:     {recall:.3f}")
        print(f"    F1 Score:   {f1:.3f}")

    print(f"\n  All figures saved to: {os.path.abspath(config.OUTPUT_DIR)}/")
    print("\n" + "#" * 64)
    print("#" + " DEMO COMPLETE ".center(62) + "#")
    print("#" * 64 + "\n")

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_demo()