"""
Tests for the simulation core functionality in the SPL onsite generator.

This module tests the Simulation class, including deterministic behavior,
data consistency, multi-domain support, and edge cases with the new
performance-level based agent system.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_simulator import Simulation, SimulationConfig, DomainSimulationSetup

# Conftest functions are automatically available in pytest
# No need to import explicitly


class TestSimulationDeterminism:
    """Test deterministic behavior of simulations."""

    def test_deterministic_with_same_seed(self, create_simulation_config):
        """Test that simulations with same seed produce identical results."""
        config1 = create_simulation_config(simulation_days=5, random_seed=42)
        config2 = create_simulation_config(simulation_days=5, random_seed=42)

        sim1 = Simulation(config1)
        sim2 = Simulation(config2)

        results1 = sim1.run()
        results2 = sim2.run()

        # Results should be identical
        pd.testing.assert_frame_equal(results1, results2)

    def test_different_with_different_seeds(self, create_simulation_config):
        """Test that simulations with different seeds produce different results."""
        config1 = create_simulation_config(simulation_days=5, random_seed=42)
        config2 = create_simulation_config(simulation_days=5, random_seed=123)

        sim1 = Simulation(config1)
        sim2 = Simulation(config2)

        results1 = sim1.run()
        results2 = sim2.run()

        # Results should be different (at least some columns)
        assert not results1.equals(results2)


class TestSimulationBasics:
    """Test basic simulation functionality."""

    def test_basic_simulation_run(self, create_simulation_config):
        """Test that a basic simulation runs without errors."""
        config = create_simulation_config(simulation_days=3, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3  # 3 days
        assert "day" in results.columns
        assert "signed_off_cumulative" in results.columns
        assert "new_tasks_created" in results.columns
        assert "avg_agent_hrs_worked_today" in results.columns  # Updated column name

    def test_multi_domain_simulation(
        self, create_domain_setup, create_simulation_config
    ):
        """Test simulation with multiple domains."""
        domains = [
            create_domain_setup("engineering", 1, 1, 0),  # Updated parameters
            create_domain_setup("law", 1, 1, 0),
        ]
        config = create_simulation_config(
            simulation_days=3, domains=domains, random_seed=42
        )
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        # Should have more activity with multiple domains
        assert results["new_tasks_created"].sum() > 0
        # Check that we have agents from both domains
        assert len(sim.agents) >= 4  # At least 2 agents per domain

    def test_minimal_simulation(self, create_domain_setup, create_simulation_config):
        """Test simulation with minimal configuration."""
        domains = [create_domain_setup("minimal", 1, 0, 0)]  # Updated parameters
        config = create_simulation_config(
            simulation_days=2, domains=domains, random_seed=42
        )
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert all(
            col in results.columns
            for col in ["day", "signed_off_cumulative", "new_tasks_created"]
        )


class TestSimulationDataConsistency:
    """Test data consistency and integrity."""

    def test_cumulative_metrics_non_decreasing(self, create_simulation_config):
        """Test that cumulative metrics never decrease."""
        config = create_simulation_config(simulation_days=10, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        # Cumulative signed off should never decrease
        signed_off = results["signed_off_cumulative"].values
        for i in range(1, len(signed_off)):
            assert (
                signed_off[i] >= signed_off[i - 1]
            ), f"Signed off decreased from {signed_off[i-1]} to {signed_off[i]} on day {i+1}"

    def test_non_negative_values(self, create_simulation_config):
        """Test that all metrics have non-negative values."""
        config = create_simulation_config(simulation_days=7, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        numeric_columns = results.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != "simulation_random_seed":  # Seed can be any integer
                assert (results[col] >= 0).all(), f"Column {col} has negative values"

    def test_day_sequence(self, create_simulation_config):
        """Test that day sequence is correct."""
        config = create_simulation_config(simulation_days=8, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        expected_days = list(range(1, 9))  # 1-indexed days
        actual_days = results["day"].tolist()
        assert actual_days == expected_days

    def test_total_tasks_consistency(self, create_simulation_config):
        """Test that total tasks in system is consistent with status counts."""
        config = create_simulation_config(simulation_days=5, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        # For each day, sum of all status counts should equal total tasks
        status_columns = [
            "tasks_claimed_eod",
            "tasks_writing_in_progress_eod",
            "tasks_needing_work_eod",
            "tasks_revision_in_progress_eod",
            "tasks_review_in_progress_eod",
            "tasks_complete_waiting_review_eod",
            "tasks_fixing_done_waiting_review_eod",
            "signed_off_cumulative",
        ]

        for _, row in results.iterrows():
            status_sum = sum(row[col] for col in status_columns if col in row)
            total_tasks = row["total_tasks_in_system"]
            assert (
                status_sum == total_tasks
            ), f"Status sum {status_sum} != total tasks {total_tasks} on day {row['day']}"


class TestSimulationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_day_simulation(self, create_simulation_config):
        """Test simulation with zero days."""
        config = create_simulation_config(simulation_days=0, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0  # No days, no data

    def test_single_day_simulation(self, create_simulation_config):
        """Test simulation with single day."""
        config = create_simulation_config(simulation_days=1, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results.iloc[0]["day"] == 1

    def test_no_top_performers(self, create_domain_setup, create_simulation_config):
        """Test simulation with no top performers (only normal contractors)."""
        domains = [create_domain_setup("test", 0, 2, 0)]  # No top performers
        config = create_simulation_config(
            simulation_days=3, domains=domains, random_seed=42
        )
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        # Should still create tasks but may have bottlenecks in review
        assert results["new_tasks_created"].sum() > 0

    def test_only_top_performers(self, create_domain_setup, create_simulation_config):
        """Test simulation with only top performers."""
        domains = [create_domain_setup("test", 2, 0, 0)]  # Only top performers
        config = create_simulation_config(
            simulation_days=3, domains=domains, random_seed=42
        )
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert results["new_tasks_created"].sum() > 0
        # Top performers should be more efficient
        final_signed_off = results["signed_off_cumulative"].iloc[-1]
        assert final_signed_off >= 0

    def test_long_simulation(self, create_simulation_config):
        """Test longer simulation runs without issues."""
        config = create_simulation_config(simulation_days=30, random_seed=42)
        sim = Simulation(config)

        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 30
        # Should see significant progress over 30 days
        assert results["signed_off_cumulative"].iloc[-1] > 0

    def test_extreme_configurations(self, create_performance_config):
        """Test with extreme performance level configurations."""
        # Very slow, low quality agents
        slow_config = create_performance_config(
            writing_hours=20.0,
            review_hours=10.0,
            average_initial_quality=0.3,
            quality_threshold=0.9,
            review_time_percentage=0.0,
        )

        # Very fast, high quality agents
        fast_config = create_performance_config(
            writing_hours=1.0,
            review_hours=0.5,
            average_initial_quality=0.95,
            quality_threshold=0.8,
            review_time_percentage=0.8,
        )

        domain = DomainSimulationSetup(
            domain_name="extreme",
            num_top_performers=1,
            num_normal_contractors=1,
            num_bad_contractors=0,
            top_performer_cfg=fast_config,
            normal_contractor_cfg=slow_config,
            bad_contractor_cfg=slow_config,  # Not used but required
        )

        config = SimulationConfig(
            simulation_days=5,
            domain_setups=[domain],
            random_seed=42,
        )

        sim = Simulation(config)
        results = sim.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        # Should still work even with extreme settings
        assert results["new_tasks_created"].sum() > 0
