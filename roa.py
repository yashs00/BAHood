"""
Region of Attraction (ROA) and Memory-Augmented Traverse Planner.

This module implements two tightly coupled components:

1. **Region of Attraction (ROA)** — For each navigation goal, the ROA
   is the set of terrain positions from which the rover can reach that
   goal without violating slope (≤ 20°) or energy-budget constraints.
   The ROA is computed via *reverse* cost-limited Dijkstra expansion on
   the terrain cost surface, starting from the goal and expanding
   outward until the energy budget is exhausted.

2. **Memory-Augmented Traverse Planner** — A path-caching layer on top
   of the NSGA-II optimizer.  Previously computed Pareto-optimal paths
   are stored alongside their ROA.  A new navigation query triggers a
   *cache hit* if and only if the rover's current position lies inside
   the ROA of an existing cached path, avoiding a costly re-optimization.

References
----------
* Lyapunov-based ROA analysis adapted for discrete terrain graphs.
* Sinha et al. (2026) — Chandrayaan-2 DFSAR ice crater context.
"""

import heapq
import numpy as np

from . import config


# =============================================================================
# TERRAIN GRAPH HELPERS
# =============================================================================

# 8-connected neighbourhood offsets (dr, dc)
_NEIGH_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def _build_edge_cost(slope, pixel_size):
    """Pre-compute per-pixel traversal cost and passability mask.

    The cost model mirrors :func:`traverse.build_cost_surfaces`:
    ``cost = pixel_size * (1 + tan(slope))``, but here we also
    return a boolean passability mask so the caller can gate
    neighbour expansion.

    Parameters
    ----------
    slope : np.ndarray, shape (R, C)
        Terrain slope in degrees.
    pixel_size : float
        Ground sampling distance (metres / pixel).

    Returns
    -------
    cost : np.ndarray, shape (R, C)
        Per-pixel traversal cost.
    passable : np.ndarray, shape (R, C), dtype bool
        True where slope ≤ ``config.ROVER_MAX_SLOPE``.
    """
    slope_rad = np.deg2rad(np.clip(slope, 0, 89))
    cost = pixel_size * (1.0 + np.tan(slope_rad))
    passable = slope <= config.ROVER_MAX_SLOPE
    return cost, passable


# =============================================================================
# REGION OF ATTRACTION  —  reverse Dijkstra
# =============================================================================

def compute_roa(goal, slope, pixel_size, energy_budget=None):
    """Compute the Region of Attraction for a navigation goal.

    Starting from *goal* the algorithm performs a **reverse** (outward)
    Dijkstra expansion.  A pixel ``(r, c)`` is included in the ROA if
    and only if:

    1. A feasible path from ``(r, c)`` to *goal* exists on the
       8-connected grid that never crosses a pixel with
       ``slope > 20°``.
    2. The cumulative traversal cost of that path does not exceed
       *energy_budget*.

    The cost at each step equals
    ``pixel_size * (1 + tan(slope))``
    (identical to the energy cost surface used by the NSGA-II planner).

    Parameters
    ----------
    goal : tuple of int
        Goal position as ``(row, col)``.
    slope : np.ndarray, shape (R, C)
        Terrain slope in degrees.
    pixel_size : float
        Ground sampling distance in metres.
    energy_budget : float, optional
        Maximum cumulative cost allowed.  Defaults to
        ``config.ROVER_BATTERY_CAPACITY`` (Wh), treating the cost
        values as an energy proxy.

    Returns
    -------
    roa_mask : np.ndarray, shape (R, C), dtype bool
        ``True`` for every pixel inside the Region of Attraction.
    cost_to_goal : np.ndarray, shape (R, C), dtype float64
        Minimum cost-to-goal from each pixel.  Pixels outside the
        ROA are set to ``np.inf``.
    """
    if energy_budget is None:
        energy_budget = config.ROVER_BATTERY_CAPACITY

    rows, cols = slope.shape
    edge_cost, passable = _build_edge_cost(slope, pixel_size)

    # Distance-to-goal array  (inf = not yet reached)
    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    dist[goal[0], goal[1]] = 0.0

    # Visited flag
    visited = np.zeros((rows, cols), dtype=bool)

    # Min-heap:  (cost_so_far, row, col)
    heap = [(0.0, goal[0], goal[1])]

    while heap:
        d, r, c = heapq.heappop(heap)

        if visited[r, c]:
            continue
        visited[r, c] = True

        # Expand to 8-connected neighbours
        for dr, dc in _NEIGH_8:
            nr, nc = r + dr, c + dc

            # Bounds check
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            # Passability check (neighbour must be traversable)
            if not passable[nr, nc]:
                continue

            # Diagonal step costs sqrt(2)× more distance
            step_scale = 1.4142135 if (dr != 0 and dc != 0) else 1.0

            # Cost of moving *from neighbour into current pixel*
            # (reverse direction, so the edge cost is at the neighbour)
            step_cost = edge_cost[nr, nc] * step_scale

            new_dist = d + step_cost

            # Budget gate
            if new_dist > energy_budget:
                continue

            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                heapq.heappush(heap, (new_dist, nr, nc))

    roa_mask = np.isfinite(dist) & (dist <= energy_budget)
    cost_to_goal = dist

    return roa_mask, cost_to_goal


# =============================================================================
# MEMORY-AUGMENTED TRAVERSE PLANNER  (path cache + ROA index)
# =============================================================================

class MemoryAugmentedPlanner:
    """Path-caching planner that uses ROA for instant query resolution.

    The planner wraps the NSGA-II optimizer from :mod:`traverse` and
    maintains a cache of previously computed paths.  Each cache entry
    stores:

    * The **goal** position.
    * The **best path** (list of waypoints).
    * The **Pareto front** objective values.
    * The **ROA mask** and **cost-to-goal** surface for the goal.

    A new ``plan(start, goal)`` call first checks whether *start*
    lies within the ROA of any cached entry with matching *goal*.
    On a hit the cached path is returned immediately.  On a miss
    the full NSGA-II optimization is executed and the result is
    cached for future queries.

    Parameters
    ----------
    slope : np.ndarray
        Terrain slope in degrees.
    hazard : np.ndarray
        Hazard map (0–1).
    illumination : np.ndarray
        Illumination fraction (0–1).
    dem : np.ndarray
        Digital Elevation Model.
    pixel_size : float
        Ground sampling distance in metres.
    energy_budget : float, optional
        Energy budget for ROA computation.
    """

    def __init__(self, slope, hazard, illumination, dem,
                 pixel_size=20.0, energy_budget=None):
        self.slope = slope
        self.hazard = hazard
        self.illumination = illumination
        self.dem = dem
        self.pixel_size = pixel_size
        self.energy_budget = (
            energy_budget if energy_budget is not None
            else config.ROVER_BATTERY_CAPACITY
        )

        # Cache: list of dicts, each representing a cached planning result
        self._cache = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def plan(self, start, goal):
        """Plan a traverse from *start* to *goal*.

        Returns cached results on ROA hit, otherwise runs NSGA-II.

        Parameters
        ----------
        start : tuple of int
            Start position ``(row, col)``.
        goal : tuple of int
            Goal position ``(row, col)``.

        Returns
        -------
        dict
            Keys:
            - ``'best_path'`` — list of waypoints.
            - ``'pareto_F'`` — Pareto front objective matrix.
            - ``'energy_profile'`` — energy profile dict.
            - ``'cache_hit'`` — bool, True if result came from cache.
            - ``'roa_mask'`` — ROA mask for the goal.
            - ``'cost_to_goal'`` — cost-to-goal surface.
        """
        # --- Check cache for ROA hit ---
        hit = self._lookup_cache(start, goal)
        if hit is not None:
            return {**hit, 'cache_hit': True}

        # --- Cache miss: run NSGA-II ---
        from .traverse import optimize_traverse

        result = optimize_traverse(
            self.dem, self.hazard, self.illumination,
            start, goal, pixel_size=self.pixel_size,
        )

        # --- Compute ROA for this goal ---
        roa_mask, cost_to_goal = compute_roa(
            goal, self.slope, self.pixel_size, self.energy_budget,
        )

        # --- Store in cache ---
        entry = {
            'goal': goal,
            'start': start,
            'best_path': result['best_path'],
            'pareto_F': result['pareto_F'],
            'energy_profile': result['energy_profile'],
            'roa_mask': roa_mask,
            'cost_to_goal': cost_to_goal,
        }
        self._cache.append(entry)

        return {**entry, 'cache_hit': False}

    def query_roa(self, position, goal):
        """Check whether *position* lies inside the ROA for *goal*.

        Parameters
        ----------
        position : tuple of int
            Candidate position ``(row, col)``.
        goal : tuple of int
            Goal position ``(row, col)``.

        Returns
        -------
        bool
            ``True`` if *position* is inside the ROA.
        float
            Cost-to-goal from *position* (``np.inf`` if outside ROA).
        """
        for entry in self._cache:
            if entry['goal'] == goal:
                r, c = position
                in_roa = bool(entry['roa_mask'][r, c])
                cost = float(entry['cost_to_goal'][r, c])
                return in_roa, cost
        # Goal not cached — compute on-the-fly
        roa_mask, cost_to_goal = compute_roa(
            goal, self.slope, self.pixel_size, self.energy_budget,
        )
        r, c = position
        return bool(roa_mask[r, c]), float(cost_to_goal[r, c])

    def invalidate(self, goal=None):
        """Clear cached entries.

        Parameters
        ----------
        goal : tuple of int, optional
            If provided, only entries for this goal are removed.
            Otherwise the entire cache is flushed.
        """
        if goal is None:
            self._cache.clear()
        else:
            self._cache = [e for e in self._cache if e['goal'] != goal]

    @property
    def cache_size(self):
        """Number of entries currently in the cache."""
        return len(self._cache)

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _lookup_cache(self, start, goal):
        """Return a cached entry whose ROA contains *start*, or None."""
        for entry in self._cache:
            if entry['goal'] != goal:
                continue
            r, c = start
            if (0 <= r < entry['roa_mask'].shape[0]
                    and 0 <= c < entry['roa_mask'].shape[1]
                    and entry['roa_mask'][r, c]):
                return entry
        return None
