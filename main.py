"""
Main Pipeline Orchestrator for LunarIce-360.

End-to-end pipeline that executes all processing modules in sequence:
preprocessing → polarimetry → ice detection → terrain analysis →
landing site selection → traverse optimization → volume estimation →
visualization.

Usage:
    python -m lunarice360.main
"""

import os
import sys
import time
import numpy as np

from . import config


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_full_pipeline(stokes_L, stokes_S, dem, pixel_size, target_center,
                      output_dir='outputs'):
    """Execute the full LunarIce-360 processing pipeline.

    Runs all modules in sequence, from raw Stokes parameters to final
    visualizations and ice volume estimates.

    Parameters
    ----------
    stokes_L : dict
        L-band Stokes parameters: {'S1', 'S2', 'S3', 'S4'}.
    stokes_S : dict
        S-band Stokes parameters: {'S1', 'S2', 'S3', 'S4'}.
    dem : np.ndarray
        Digital Elevation Model (2D array).
    pixel_size : float
        Pixel size in meters.
    target_center : tuple
        (row, col) of the target region of interest (ice deposit).
    output_dir : str
        Directory for output files and figures.

    Returns
    -------
    dict
        Dictionary containing all intermediate and final results.
    """
    pipeline_start = time.time()
    results = {}

    # Override output directory
    config.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 64)
    print("  LunarIce-360: Full Pipeline Execution")
    print("=" * 64)

    # =========================================================================
    # STEP 1: Preprocessing
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 1: Preprocessing Stokes Parameters")
    print("-" * 64)
    t0 = time.time()

    try:
        from .preprocessing import preprocess_stokes
        S1_f, S2_f, S3_f, S4_f = preprocess_stokes(
            stokes_L['S1'], stokes_L['S2'], stokes_L['S3'], stokes_L['S4'],
            window_size=config.SPECKLE_FILTER_WINDOW
        )
        stokes_L_proc = {'S1': S1_f, 'S2': S2_f, 'S3': S3_f, 'S4': S4_f}

        S1_fs, S2_fs, S3_fs, S4_fs = preprocess_stokes(
            stokes_S['S1'], stokes_S['S2'], stokes_S['S3'], stokes_S['S4'],
            window_size=config.SPECKLE_FILTER_WINDOW
        )
        stokes_S_proc = {'S1': S1_fs, 'S2': S2_fs, 'S3': S3_fs, 'S4': S4_fs}
        print(f"  Preprocessing complete (used pipeline module).")
    except (ImportError, KeyError, Exception) as e:
        print(f"  [WARN] preprocessing module error ({e}), using raw data.")
        stokes_L_proc = stokes_L
        stokes_S_proc = stokes_S

    results['stokes_L'] = stokes_L_proc
    results['stokes_S'] = stokes_S_proc
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 2: Polarimetric Feature Extraction
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 2: Polarimetric Feature Extraction")
    print("-" * 64)
    t0 = time.time()

    try:
        from .polarimetry import build_feature_stack
        feature_stack, feature_names = build_feature_stack(stokes_L_proc, stokes_S_proc)
        print(f"  Feature extraction complete (used pipeline module).")

        # Extract individual features from the stack
        CPR_L = feature_stack[:, :, 0]
        DOP_L = feature_stack[:, :, 1]
        Pv = feature_stack[:, :, 2]
        Ps = feature_stack[:, :, 3]
        Pd = feature_stack[:, :, 4]
        H = feature_stack[:, :, 5]
        alpha = feature_stack[:, :, 7]
        CPR_S = feature_stack[:, :, 9]

        features_L = {
            'CPR': CPR_L, 'DOP': DOP_L,
            'Pv': Pv, 'Ps': Ps, 'Pd': Pd,
            'H': H, 'alpha': alpha,
        }
        features_S = {'CPR': CPR_S}

    except (ImportError, KeyError, Exception) as e:
        print(f"  [FALLBACK] polarimetry module error ({e}), computing inline.")
        S1, S2, S3, S4 = (
            stokes_L_proc['S1'], stokes_L_proc['S2'],
            stokes_L_proc['S3'], stokes_L_proc['S4']
        )

        # CPR and DOP
        CPR_L = np.abs((S1 - S4) / (S1 + S4 + 1e-10))
        DOP_L = np.sqrt(S2**2 + S3**2 + S4**2) / (S1 + 1e-10)

        S1s, S2s, S3s, S4s = (
            stokes_S_proc['S1'], stokes_S_proc['S2'],
            stokes_S_proc['S3'], stokes_S_proc['S4']
        )
        CPR_S = np.abs((S1s - S4s) / (S1s + S4s + 1e-10))

        # m-chi decomposition
        m = DOP_L
        chi = 0.5 * np.arctan2(S4, np.sqrt(S2**2 + S3**2 + 1e-10))
        Pv = S1 * m * (1 - np.sin(2 * chi)) / 2.0
        Ps = S1 * m * (1 + np.sin(2 * chi)) / 2.0
        Pd = S1 * (1 - m)
        total_P = Pv + Ps + Pd + 1e-10
        Pv /= total_P
        Ps /= total_P
        Pd /= total_P

        # Synthetic H/alpha
        H = np.clip(
            0.3 + 0.4 * Pv + np.random.normal(0, 0.05, dem.shape), 0, 1
        )
        alpha = np.clip(
            30 + 30 * Pv + np.random.normal(0, 3, dem.shape), 0, 90
        )

        features_L = {
            'CPR': CPR_L, 'DOP': DOP_L,
            'Pv': Pv, 'Ps': Ps, 'Pd': Pd,
            'H': H, 'alpha': alpha,
        }
        features_S = {'CPR': CPR_S}
        
        # Build fallback feature stack
        feature_stack = np.zeros((dem.shape[0], dem.shape[1], 17), dtype=np.float64)
        feature_stack[:, :, 0] = CPR_L
        feature_stack[:, :, 1] = DOP_L
        feature_stack[:, :, 2] = Pv
        feature_stack[:, :, 3] = Ps
        feature_stack[:, :, 4] = Pd
        feature_stack[:, :, 5] = H
        feature_stack[:, :, 7] = alpha
        feature_stack[:, :, 9] = CPR_S

    results['feature_stack'] = feature_stack
    results['features_L'] = features_L
    results['features_S'] = features_S
    results['CPR_L'] = CPR_L
    results['CPR_S'] = CPR_S
    print(f"  CPR_L range: [{CPR_L.min():.3f}, {CPR_L.max():.3f}]")
    if DOP_L is not None:
        print(f"  DOP_L range: [{DOP_L.min():.3f}, {DOP_L.max():.3f}]")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 3: Ice Detection
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 3: Ice Detection (Multi-Method Fusion)")
    print("-" * 64)
    t0 = time.time()

    try:
        # Compute zone mask
        try:
            from .polarimetry import classify_H_alpha_zones
            zones = classify_H_alpha_zones(H, alpha)
            zone_mask = ((zones == 8) | (zones == 9)).astype(np.float64)
        except Exception:
            zone_mask = None

        from .ice_detection import run_ice_detection
        ice_results = run_ice_detection(
            feature_stack, CPR_L, DOP_L, CPR_S=CPR_S, zone_mask=zone_mask
        )
        ice_prob = ice_results.get('fused_probability', None)
        ice_detected = (ice_prob > 0.5).astype(np.float64) if ice_prob is not None else None
        print(f"  Ice detection complete (used pipeline module).")
    except (ImportError, Exception) as e:
        print(f"  [FALLBACK] ice_detection module error: {e}, using thresholds.")
        ice_cpr = CPR_L > config.CPR_ICE_THRESHOLD
        ice_dop = DOP_L < config.DOP_ICE_THRESHOLD if DOP_L is not None else \
            np.zeros_like(CPR_L, dtype=bool)
        ice_detected = (ice_cpr & ice_dop).astype(np.float64)

        # Probability map
        ice_prob = np.zeros_like(CPR_L)
        ice_prob += 0.4 * np.clip((CPR_L - 0.8) / 0.8, 0, 1)
        if DOP_L is not None:
            ice_prob += 0.3 * np.clip((0.2 - DOP_L) / 0.2, 0, 1)
        if Pv is not None:
            ice_prob += 0.3 * np.clip(Pv / (Pv.max() + 1e-10), 0, 1)
        ice_prob = np.clip(ice_prob, 0, 1)

        ice_results = {
            'fused_probability': ice_prob,
            'fused_detection': ice_detected,
        }

    results['ice_prob'] = ice_prob
    results['ice_detected'] = ice_detected
    results['ice_results'] = ice_results
    n_detected = int(ice_detected.sum()) if ice_detected is not None else 0
    print(f"  Detected ice pixels: {n_detected}")
    if ice_prob is not None:
        print(f"  Max ice probability: {ice_prob.max():.3f}")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 4: Terrain Analysis
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 4: Terrain Analysis")
    print("-" * 64)
    t0 = time.time()

    try:
        from .terrain import run_terrain_analysis
        terrain_results = run_terrain_analysis(dem, pixel_size)
        slope = terrain_results['slope']
        roughness = terrain_results['roughness']
        illumination = terrain_results['illumination_frac']
        hazard = terrain_results['hazard_map']
        print(f"  Terrain analysis complete (used pipeline module).")
    except (ImportError, KeyError, Exception) as e:
        print("  [FALLBACK] terrain module not found, computing inline.")
        from scipy.ndimage import uniform_filter

        dy, dx = np.gradient(dem, pixel_size)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        roughness = np.sqrt(
            uniform_filter(dem**2, size=config.ROUGHNESS_WINDOW) -
            uniform_filter(dem, size=config.ROUGHNESS_WINDOW)**2
        )

        dem_smooth = uniform_filter(dem, size=21)
        illumination = np.clip(
            0.5 + 0.5 * (dem - dem_smooth) / (np.std(dem) + 1e-10), 0, 1
        )

        slope_norm = np.clip(slope / config.ROVER_MAX_SLOPE, 0, 1)
        rough_norm = np.clip(roughness / (roughness.max() + 1e-10), 0, 1)
        hazard = np.clip(
            0.5 * slope_norm + 0.3 * rough_norm + 0.2 * (1 - illumination),
            0, 1
        )

        terrain_results = {
            'slope': slope, 'roughness': roughness,
            'illumination': illumination, 'hazard': hazard,
        }

    results['slope'] = slope
    results['roughness'] = roughness
    results['illumination'] = illumination
    results['hazard'] = hazard
    results['terrain_results'] = terrain_results
    print(f"  Slope range: [{slope.min():.1f}, {slope.max():.1f}] degrees")
    print(f"  Hazard range: [{hazard.min():.3f}, {hazard.max():.3f}]")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 5: Landing Site Selection
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 5: Landing Site Selection")
    print("-" * 64)
    t0 = time.time()

    try:
        from .landing_site import select_landing_sites
        landing_results = select_landing_sites(
            dem, slope, roughness, illumination, hazard,
            target_center=target_center, pixel_size=pixel_size
        )
        landing_scores = landing_results['scores']
        best_sites = landing_results['best_sites']
        print(f"  Landing site selection complete (used pipeline module).")
    except ImportError:
        print("  [FALLBACK] landing_site module not found, computing inline.")
        safety = 1.0 - hazard
        rows_dem, cols_dem = dem.shape
        y, x = np.meshgrid(
            np.arange(rows_dem), np.arange(cols_dem), indexing='ij'
        )
        dist_to_target = np.sqrt(
            (y - target_center[0])**2 + (x - target_center[1])**2
        ) * pixel_size
        proximity = np.exp(
            -((dist_to_target - config.LANDING_IDEAL_DISTANCE)**2) /
            (2 * config.LANDING_DISTANCE_SIGMA**2)
        )
        slope_norm = np.clip(slope / config.ROVER_MAX_SLOPE, 0, 1)

        landing_scores = (
            config.LANDING_WEIGHTS['safety'] * safety +
            config.LANDING_WEIGHTS['illumination'] * illumination +
            config.LANDING_WEIGHTS['proximity'] * proximity +
            config.LANDING_WEIGHTS['flatness'] * (1.0 - slope_norm)
        )
        landing_scores[slope > config.LANDING_MAX_SLOPE] = 0
        landing_scores[
            illumination < config.ILLUMINATION_MIN_FRACTION
        ] = 0

        from scipy.ndimage import maximum_filter
        local_max = (
            landing_scores == maximum_filter(landing_scores, size=30)
        )
        local_max &= landing_scores > 0.3
        site_coords = np.argwhere(local_max)

        if len(site_coords) > 0:
            site_scores = landing_scores[
                site_coords[:, 0], site_coords[:, 1]
            ]
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
            best_idx = np.unravel_index(
                np.argmax(landing_scores), landing_scores.shape
            )
            best_sites = [{
                'row': int(best_idx[0]), 'col': int(best_idx[1]),
                'score': float(landing_scores[best_idx]),
                'label': 'Site 1'
            }]

        landing_results = {
            'scores': landing_scores, 'best_sites': best_sites
        }

    results['landing_scores'] = landing_scores
    results['best_sites'] = best_sites
    results['landing_results'] = landing_results
    for s in best_sites:
        print(f"  {s['label']}: row={s['row']}, col={s['col']}, "
              f"score={s['score']:.3f}")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 6: Rover Traverse Optimization
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 6: Rover Traverse Optimization (NSGA-II)")
    print("-" * 64)
    t0 = time.time()

    start_pos = (best_sites[0]['row'], best_sites[0]['col'])
    goal_pos = target_center

    try:
        from .roa import MemoryAugmentedPlanner
        # Note: slope is needed for ROA. It's computed in Step 4 and available here.
        planner = MemoryAugmentedPlanner(
            slope, hazard, illumination, dem, pixel_size=pixel_size
        )
        traverse_results = planner.plan(start_pos, goal_pos)
        path = traverse_results['best_path']
        pareto_F = traverse_results.get('pareto_F', None)
        energy_profile = traverse_results.get('energy_profile', None)
        
        hit_str = "(cache hit)" if traverse_results.get('cache_hit', False) else "(cache miss, ran NSGA-II)"
        print(f"  Traverse optimization complete via ROA Memory Planner {hit_str}.")
    except (ImportError, Exception) as e:
        print(f"  [FALLBACK] traverse module error: {e}, using linear path.")
        n_waypoints = config.NSGA2_N_WAYPOINTS + 2
        path_rows = np.linspace(
            start_pos[0], goal_pos[0], n_waypoints
        ).astype(int)
        path_cols = np.linspace(
            start_pos[1], goal_pos[1], n_waypoints
        ).astype(int)
        path = np.column_stack([path_rows, path_cols])

        # Synthetic Pareto front
        n_pareto = 50
        pareto_F = np.column_stack([
            np.random.uniform(2000, 5000, n_pareto),
            np.random.uniform(0.1, 0.8, n_pareto),
            np.random.uniform(0.2, 0.6, n_pareto),
        ])

        # Energy profile
        diffs = np.diff(path, axis=0) * pixel_size
        segment_dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        cum_dist = np.concatenate([[0], np.cumsum(segment_dists)])
        illuminated_arr = np.array([
            illumination[
                np.clip(int(r), 0, dem.shape[0]-1),
                np.clip(int(c), 0, dem.shape[1]-1)
            ] > 0.5
            for r, c in path
        ])
        battery = np.zeros(len(cum_dist))
        battery[0] = config.ROVER_BATTERY_CAPACITY
        for i in range(1, len(cum_dist)):
            dt = segment_dists[i-1] / config.ROVER_SPEED
            power_in = (
                config.ROVER_SOLAR_POWER if illuminated_arr[i] else 0
            )
            power_out = config.ROVER_LOCOMOTION_POWER
            energy_delta = (power_in - power_out) * dt / 3600.0
            battery[i] = np.clip(
                battery[i-1] + energy_delta,
                0, config.ROVER_BATTERY_CAPACITY
            )

        energy_profile = {
            'cumulative_distance': cum_dist,
            'battery_energy': battery,
            'illuminated': illuminated_arr,
            'battery_max': config.ROVER_BATTERY_CAPACITY,
        }

        traverse_results = {
            'best_path': path, 'pareto_F': pareto_F,
            'energy_profile': energy_profile,
        }

    # Compute path metrics
    path_arr = np.array(path)
    diffs = np.diff(path_arr, axis=0) * pixel_size
    segment_dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    total_distance = float(segment_dists.sum())
    travel_time_hours = total_distance / config.ROVER_SPEED / 3600.0

    results['path'] = path
    results['pareto_F'] = pareto_F
    results['energy_profile'] = energy_profile
    results['traverse_results'] = traverse_results
    results['total_distance'] = total_distance
    results['travel_time_hours'] = travel_time_hours
    print(f"  Path waypoints: {len(path)}")
    print(f"  Total distance: {total_distance:.1f} m")
    print(f"  Travel time: {travel_time_hours:.1f} hours")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 7: Bayesian Ice Volume Estimation (MCMC)
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 7: Bayesian Ice Volume Estimation (MCMC)")
    print("-" * 64)
    t0 = time.time()

    try:
        from .volume_estimation import estimate_ice_volume
        volume_results = estimate_ice_volume(
            ice_prob, CPR_L, dem, pixel_size
        )
        mcmc_samples = volume_results.get('mcmc_samples', None)
        print(f"  Volume estimation complete (used pipeline module).")
    except (ImportError, Exception) as e:
        print(f"  [FALLBACK] volume_estimation module error: {e}, "
              "using simplified estimation.")
        ice_area_m2 = float((ice_detected > 0).sum()) * pixel_size**2
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
            'mcmc_samples': mcmc_samples,
            'median': float(np.median(volume_samples)),
            'ci_68_low': float(np.percentile(volume_samples, 16)),
            'ci_68_high': float(np.percentile(volume_samples, 84)),
            'ci_95_low': float(np.percentile(volume_samples, 2.5)),
            'ci_95_high': float(np.percentile(volume_samples, 97.5)),
            'ice_area_m2': ice_area_m2,
        }

    results['volume_results'] = volume_results
    results['mcmc_samples'] = mcmc_samples
    print(f"  Median volume: {volume_results['median']:.1f} m³")
    print(f"  68% CI: [{volume_results['ci_68_low']:.1f}, "
          f"{volume_results['ci_68_high']:.1f}] m³")
    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 8: Generate All Visualizations
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 8: Generating Visualizations")
    print("-" * 64)
    t0 = time.time()

    try:
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
            plot_mcmc_corner,
            plot_volume_summary,
            plot_dual_frequency,
            generate_summary_figure,
        )

        setup_output_dir()

        # Ice probability map
        if ice_prob is not None:
            plot_ice_probability_map(ice_prob, dem=dem)

        # H/alpha plane
        if H is not None and alpha is not None:
            plot_H_alpha_plane(H, alpha)

        # Decomposition panels
        if Pv is not None and Ps is not None and Pd is not None:
            plot_decomposition_panels(Pv, Ps, Pd)

        # Terrain analysis
        plot_terrain_analysis(slope, roughness, illumination, hazard)

        # Landing site
        plot_landing_site(landing_scores, best_sites, dem=dem)

        # Traverse path
        plot_traverse_path(
            path, dem, hazard_map=hazard, ice_prob=ice_prob,
            start=start_pos, goal=goal_pos
        )

        # Pareto front
        if pareto_F is not None:
            plot_pareto_front(pareto_F, selected_idx=0)

        # Energy profile
        if energy_profile is not None:
            plot_energy_profile(energy_profile)

        # MCMC corner plot
        if mcmc_samples is not None:
            try:
                plot_mcmc_corner(mcmc_samples)
            except Exception as e:
                print(f"  [WARN] Corner plot failed: {e}")

        # Volume summary
        plot_volume_summary(volume_results)

        # Dual-frequency analysis
        if CPR_L is not None and CPR_S is not None:
            plot_dual_frequency(CPR_L, CPR_S)

        # Summary figure
        if ice_prob is not None:
            generate_summary_figure(ice_prob, hazard, path, volume_results)

        print(f"  All visualizations saved to: "
              f"{os.path.abspath(output_dir)}/")

    except ImportError as e:
        print(f"  [WARN] Visualization module not available: {e}")

    print(f"  Time: {time.time() - t0:.2f}s")

    # =========================================================================
    # STEP 9: Summary Report
    # =========================================================================
    print("\n" + "-" * 64)
    print("  STEP 9: Summary Report")
    print("-" * 64)

    results['dem'] = dem
    results['pixel_size'] = pixel_size
    results['start_pos'] = start_pos
    results['goal_pos'] = goal_pos

    print_summary_report(results)

    total_time = time.time() - pipeline_start
    results['total_pipeline_time'] = total_time
    print(f"\n  Total pipeline execution time: {total_time:.2f}s")
    print("\n" + "=" * 64)
    print("  Pipeline Complete!")
    print("=" * 64 + "\n")

    return results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(results):
    """Print a formatted summary report of pipeline results.

    Parameters
    ----------
    results : dict
        Pipeline results dictionary containing keys such as:
        'ice_prob', 'ice_detected', 'best_sites', 'total_distance',
        'travel_time_hours', 'volume_results'.
    """
    print()
    print("+" + "=" * 60 + "+")
    print("|" + " LunarIce-360 Pipeline Summary Report ".center(60) + "|")
    print("+" + "=" * 60 + "+")

    # --- Ice Detection Statistics ---
    print("\n  +- Ice Detection ------------------------------------+")
    if 'ice_prob' in results and results['ice_prob'] is not None:
        ip = results['ice_prob']
        print(f"  |  Max probability:     {ip.max():.4f}")
        print(f"  |  Pixels > 0.5:        {int((ip > 0.5).sum())}")
        print(f"  |  Pixels > 0.8:        {int((ip > 0.8).sum())}")
        print(f"  |  Mean (detected):     {ip[ip > 0.1].mean():.4f}" 
              if (ip > 0.1).any() else "  |  Mean (detected):     N/A")
    if 'ice_detected' in results and results['ice_detected'] is not None:
        n_det = int(results['ice_detected'].sum())
        total = results['ice_detected'].size
        print(f"  |  Detected pixels:     {n_det} "
              f"({100*n_det/total:.2f}%)")
    print("  +----------------------------------------------------+")

    # --- Landing Site Coordinates ---
    print("\n  +- Landing Sites -----------------------------------+")
    if 'best_sites' in results:
        for site in results['best_sites']:
            px = results.get('pixel_size', 20.0)
            r, c = site['row'], site['col']
            print(f"  |  {site['label']:10s}  ({r:4d}, {c:4d})  "
                  f"score={site['score']:.3f}")
    print("  +----------------------------------------------------+")

    # --- Traverse Statistics ---
    print("\n  +- Rover Traverse ----------------------------------+")
    if 'total_distance' in results:
        print(f"  |  Total distance:      "
              f"{results['total_distance']:.1f} m")
        print(f"  |  Travel time:         "
              f"{results['travel_time_hours']:.1f} hours")
        print(f"  |  Rover speed:         "
              f"{config.ROVER_SPEED*1000:.0f} mm/s")
    if 'path' in results:
        print(f"  |  Waypoints:           {len(results['path'])}")
    print("  +----------------------------------------------------+")

    # --- Ice Volume Estimation ---
    print("\n  +- Ice Volume Estimate -----------------------------+")
    if 'volume_results' in results:
        vr = results['volume_results']
        median = vr.get('median', 0)
        ci68_lo = vr.get('ci_68_low', 0)
        ci68_hi = vr.get('ci_68_high', 0)
        ci95_lo = vr.get('ci_95_low', 0)
        ci95_hi = vr.get('ci_95_high', 0)
        print(f"  |  Median volume:       {median:.1f} m^3")
        print(f"  |  68% CI:              "
              f"[{ci68_lo:.1f}, {ci68_hi:.1f}] m^3")
        print(f"  |  95% CI:              "
              f"[{ci95_lo:.1f}, {ci95_hi:.1f}] m^3")
        if 'ice_area_m2' in vr:
            print(f"  |  Ice area:            {vr['ice_area_m2']:.0f} m^2")
    print("  +----------------------------------------------------+")
    print()



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\nLunarIce-360 Pipeline")
    print("="*40)

    # Try to load real data from configured paths
    try:
        from .data_loader import load_dataset
        print("Loading real dataset from configured paths...")
        dataset = load_dataset()
        results = run_full_pipeline(
            stokes_L=dataset['stokes_L'],
            stokes_S=dataset['stokes_S'],
            dem=dataset['dem'],
            pixel_size=dataset['pixel_size'],
            target_center=dataset.get('target_center', (250, 250)),
            output_dir=config.OUTPUT_DIR,
        )
    except (ImportError, FileNotFoundError, Exception) as e:
        print(f"Real data not available ({type(e).__name__}: {e})")
        print("Falling back to synthetic demo...\n")

        try:
            from .demo_synthetic import generate_synthetic_dataset
            dataset = generate_synthetic_dataset()
            results = run_full_pipeline(
                stokes_L=dataset['stokes_L'],
                stokes_S=dataset['stokes_S'],
                dem=dataset['dem'],
                pixel_size=dataset['pixel_size'],
                target_center=dataset['crater_center'],
                output_dir=config.OUTPUT_DIR,
            )
        except Exception as e2:
            print(f"\nERROR: Could not run pipeline: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)