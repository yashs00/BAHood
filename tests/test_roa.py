"""
Tests for the Region of Attraction (ROA) and Memory-Augmented Planner.

Tests cover:
    1. Basic ROA computation on a flat, fully-passable grid.
    2. Slope barrier blocking ROA expansion.
    3. Energy budget limiting ROA extent.
    4. Goal pixel is always inside its own ROA.
    5. MemoryAugmentedPlanner cache-hit / cache-miss logic.
    6. query_roa returns correct in-ROA / out-of-ROA results.
    7. Cache invalidation.
    8. Diagonal cost scaling (√2 factor).
"""

import os
import sys
import numpy as np
import pytest

# Ensure the project root is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_terrain():
    """50×50 grid with zero slope (fully passable, uniform cost)."""
    rows, cols = 50, 50
    slope = np.zeros((rows, cols), dtype=np.float64)
    pixel_size = 20.0
    return slope, pixel_size


@pytest.fixture
def barrier_terrain():
    """50×50 grid with a vertical wall of slope=30° cutting the grid in half."""
    rows, cols = 50, 50
    slope = np.zeros((rows, cols), dtype=np.float64)
    # Impassable wall at column 25
    slope[:, 25] = 30.0
    pixel_size = 20.0
    return slope, pixel_size


@pytest.fixture
def simple_dem():
    """50×50 flat DEM at 0 m elevation."""
    return np.zeros((50, 50), dtype=np.float64)


@pytest.fixture
def simple_hazard():
    """50×50 zero-hazard map."""
    return np.zeros((50, 50), dtype=np.float64)


@pytest.fixture
def simple_illumination():
    """50×50 fully-illuminated map."""
    return np.ones((50, 50), dtype=np.float64)


# ---------------------------------------------------------------------------
# compute_roa tests
# ---------------------------------------------------------------------------

class TestComputeROA:
    """Tests for roa.compute_roa."""

    def test_goal_always_in_roa(self, flat_terrain):
        """The goal pixel must always be inside its own ROA."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        roa_mask, cost_to_goal = compute_roa(goal, slope, pixel_size)

        assert roa_mask[goal[0], goal[1]], "Goal must be in its own ROA"
        assert cost_to_goal[goal[0], goal[1]] == 0.0, "Cost at goal must be 0"

    def test_flat_terrain_full_coverage(self, flat_terrain):
        """On a flat grid with large budget, ROA should cover everything."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        roa_mask, cost_to_goal = compute_roa(
            goal, slope, pixel_size, energy_budget=1e9,
        )

        # Every pixel should be reachable
        assert roa_mask.all(), (
            f"Expected full coverage, but {(~roa_mask).sum()} pixels are outside ROA"
        )

    def test_cost_increases_with_distance(self, flat_terrain):
        """Cost-to-goal should monotonically increase with grid distance."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        _, cost_to_goal = compute_roa(goal, slope, pixel_size, energy_budget=1e9)

        # Immediate neighbours should cost less than far corners
        cost_neighbour = cost_to_goal[25, 26]
        cost_corner = cost_to_goal[0, 0]
        assert cost_neighbour < cost_corner, (
            f"Neighbour cost ({cost_neighbour:.1f}) should be less than "
            f"corner cost ({cost_corner:.1f})"
        )

    def test_barrier_blocks_expansion(self, barrier_terrain):
        """A slope wall should prevent ROA from crossing to the other side."""
        from BAHood.roa import compute_roa
        slope, pixel_size = barrier_terrain
        goal = (25, 10)  # left side of the wall
        roa_mask, _ = compute_roa(goal, slope, pixel_size, energy_budget=1e9)

        # Pixels on the right side of the wall should NOT be in ROA
        assert not roa_mask[25, 30], "Pixel behind slope wall should be outside ROA"
        # Pixels on the left side should be in ROA
        assert roa_mask[25, 10], "Pixel on the same side as goal should be in ROA"

    def test_small_energy_budget_limits_roa(self, flat_terrain):
        """A very small energy budget should produce a tiny ROA."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        # pixel_size=20, cost per step on flat = 20*(1+tan(0))=20
        # Budget of 60 allows at most ~3 steps from goal
        roa_mask, _ = compute_roa(goal, slope, pixel_size, energy_budget=60.0)

        n_in_roa = roa_mask.sum()
        total_pixels = slope.shape[0] * slope.shape[1]
        assert n_in_roa < total_pixels, (
            "With tiny budget, ROA should not cover entire grid"
        )
        assert n_in_roa > 0, "ROA must contain at least the goal pixel"
        assert roa_mask[25, 25], "Goal must always be in ROA"

    def test_diagonal_costs_sqrt2(self, flat_terrain):
        """Diagonal neighbours should have √2 × higher cost than cardinal."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        _, cost_to_goal = compute_roa(goal, slope, pixel_size, energy_budget=1e9)

        # Cardinal neighbour (one step right)
        cost_cardinal = cost_to_goal[25, 26]
        # Diagonal neighbour (one step down-right)
        cost_diagonal = cost_to_goal[26, 26]

        expected_ratio = np.sqrt(2)
        actual_ratio = cost_diagonal / cost_cardinal
        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"Diagonal/cardinal cost ratio should be √2≈{expected_ratio:.4f}, "
            f"got {actual_ratio:.4f}"
        )

    def test_roa_symmetry_on_flat(self, flat_terrain):
        """On a flat, uniform grid the ROA should be symmetric about the goal."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (25, 25)
        _, cost_to_goal = compute_roa(goal, slope, pixel_size, energy_budget=1e9)

        # Cost should be equal for pixels equidistant from goal
        assert abs(cost_to_goal[25, 26] - cost_to_goal[25, 24]) < 1e-10
        assert abs(cost_to_goal[26, 25] - cost_to_goal[24, 25]) < 1e-10
        assert abs(cost_to_goal[26, 26] - cost_to_goal[24, 24]) < 1e-10

    def test_edge_goal(self, flat_terrain):
        """ROA should work when the goal is at the grid edge."""
        from BAHood.roa import compute_roa
        slope, pixel_size = flat_terrain
        goal = (0, 0)
        roa_mask, cost_to_goal = compute_roa(
            goal, slope, pixel_size, energy_budget=1e9,
        )
        assert roa_mask[goal[0], goal[1]]
        assert cost_to_goal[goal[0], goal[1]] == 0.0

    def test_steep_terrain_excludes_pixels(self):
        """Pixels with slope > 20° should never appear in the ROA."""
        from BAHood.roa import compute_roa
        slope = np.full((20, 20), 25.0)   # everything too steep
        slope[10, 10] = 5.0               # except the goal
        pixel_size = 20.0
        goal = (10, 10)
        roa_mask, _ = compute_roa(goal, slope, pixel_size, energy_budget=1e9)

        # Only the goal itself should be in the ROA
        assert roa_mask.sum() == 1, (
            f"Only goal should be reachable, but {roa_mask.sum()} pixels are in ROA"
        )


# ---------------------------------------------------------------------------
# MemoryAugmentedPlanner tests
# ---------------------------------------------------------------------------

class TestMemoryAugmentedPlanner:
    """Tests for roa.MemoryAugmentedPlanner."""

    def test_first_call_is_cache_miss(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """First plan() call should be a cache miss."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size,
        )
        result = planner.plan(start=(5, 5), goal=(25, 25))
        assert result['cache_hit'] is False
        assert planner.cache_size == 1

    def test_second_call_is_cache_hit(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """Repeated plan() to the same goal from a position in the ROA
        should be a cache hit."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size, energy_budget=1e9,
        )
        # First call: cache miss
        planner.plan(start=(5, 5), goal=(25, 25))
        # Second call from a nearby position: should hit cache
        result2 = planner.plan(start=(10, 10), goal=(25, 25))
        assert result2['cache_hit'] is True
        assert planner.cache_size == 1  # no new entry added

    def test_different_goal_is_cache_miss(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """Planning to a different goal should be a cache miss."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size,
        )
        planner.plan(start=(5, 5), goal=(25, 25))
        result2 = planner.plan(start=(5, 5), goal=(40, 40))
        assert result2['cache_hit'] is False
        assert planner.cache_size == 2

    def test_invalidate_clears_cache(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """invalidate() should clear the cache."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size,
        )
        planner.plan(start=(5, 5), goal=(25, 25))
        assert planner.cache_size == 1
        planner.invalidate()
        assert planner.cache_size == 0

    def test_invalidate_specific_goal(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """invalidate(goal) should only remove entries for that goal."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size,
        )
        planner.plan(start=(5, 5), goal=(25, 25))
        planner.plan(start=(5, 5), goal=(40, 40))
        assert planner.cache_size == 2
        planner.invalidate(goal=(25, 25))
        assert planner.cache_size == 1

    def test_query_roa_in_roa(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """query_roa should return True for positions inside ROA."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size, energy_budget=1e9,
        )
        planner.plan(start=(5, 5), goal=(25, 25))
        in_roa, cost = planner.query_roa((10, 10), (25, 25))
        assert in_roa is True
        assert cost < np.inf

    def test_query_roa_out_of_roa(self, barrier_terrain, simple_dem,
                                   simple_hazard, simple_illumination):
        """query_roa should return False for positions blocked by barrier."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = barrier_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size, energy_budget=1e9,
        )
        planner.plan(start=(5, 5), goal=(25, 10))  # goal on left of wall
        in_roa, cost = planner.query_roa((25, 30), (25, 10))  # right of wall
        assert in_roa is False
        assert cost == np.inf

    def test_result_contains_roa_mask(
        self, flat_terrain, simple_dem, simple_hazard, simple_illumination,
    ):
        """Plan result should contain roa_mask and cost_to_goal."""
        from BAHood.roa import MemoryAugmentedPlanner
        slope, pixel_size = flat_terrain
        planner = MemoryAugmentedPlanner(
            slope, simple_hazard, simple_illumination, simple_dem,
            pixel_size=pixel_size,
        )
        result = planner.plan(start=(5, 5), goal=(25, 25))
        assert 'roa_mask' in result
        assert 'cost_to_goal' in result
        assert result['roa_mask'].shape == slope.shape
        assert result['cost_to_goal'].shape == slope.shape
