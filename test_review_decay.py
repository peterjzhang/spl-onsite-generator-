#!/usr/bin/env python3
"""Test script to verify review time decay functionality."""

import task_simulator
import numpy as np


def test_review_time_decay():
    """Test that review time decreases when reviewing the same agent multiple times."""

    # Create a performance config for a reviewer with decay factor 0.8 (20% faster each time)
    reviewer_config = task_simulator.PerformanceLevelConfig(
        review_hours=4.0,
        review_hours_noise=0.0,  # No noise for predictable testing
        review_time_decay=0.8,
    )

    # Set seed for reproducible results
    np.random.seed(42)

    reviewer = task_simulator.Agent(
        id="TestReviewer",
        domain_name="Test",
        performance_level="top_performer",
        cfg=reviewer_config,
    )

    # Test effective review time for multiple reviews of same trainer
    trainer_id = "TestTrainer"

    print("Review Time Decay Test:")
    print(f"Base review time: {reviewer.actual_review_hours:.2f} hours")
    print(f"Decay factor: {reviewer_config.review_time_decay}")
    print()

    # Simulate multiple reviews
    for review_num in range(1, 6):
        # Get effective time before incrementing count
        effective_time = reviewer.get_effective_review_time(trainer_id)
        expected_time = reviewer.actual_review_hours * (
            reviewer_config.review_time_decay ** (review_num - 1)
        )

        print(f"Review #{review_num}:")
        print(f"  Expected time: {expected_time:.2f} hours")
        print(f"  Actual time:   {effective_time:.2f} hours")
        print(
            f"  Time savings:  {(1 - effective_time/reviewer.actual_review_hours)*100:.1f}%"
        )

        # Manually increment count for next iteration
        reviewer.trainer_review_counts[trainer_id] = (
            reviewer.trainer_review_counts.get(trainer_id, 0) + 1
        )
        print()


def test_deterministic_with_decay():
    """Test that simulations with decay remain deterministic."""

    # Create performance level configs
    top_performer_cfg = task_simulator.PerformanceLevelConfig(
        review_time_decay=0.85, review_time_percentage=0.8
    )
    normal_contractor_cfg = task_simulator.PerformanceLevelConfig(
        review_time_percentage=0.0
    )
    bad_contractor_cfg = task_simulator.PerformanceLevelConfig(
        review_time_percentage=0.0
    )

    config = task_simulator.SimulationConfig(
        simulation_days=5,
        domain_setups=[
            task_simulator.DomainSimulationSetup(
                domain_name="Test",
                num_top_performers=2,
                num_normal_contractors=2,
                num_bad_contractors=1,
                top_performer_cfg=top_performer_cfg,
                normal_contractor_cfg=normal_contractor_cfg,
                bad_contractor_cfg=bad_contractor_cfg,
            )
        ],
        random_seed=123,
    )

    # Run simulation twice
    results = []
    for run in range(2):
        sim = task_simulator.Simulation(config=config)
        df = sim.run()

        metrics = {
            "tasks_decisioned": df["tasks_decisioned_by_review"].sum(),
            "signed_off": df["signed_off_cumulative"].iloc[-1],
            "avg_quality": df["avg_quality_signed_off"].iloc[-1],
        }
        results.append(metrics)

    print("Determinism test with review decay:")
    for key in results[0]:
        if results[0][key] == results[1][key]:
            print(f"  ✓ {key}: IDENTICAL")
        else:
            print(f"  ✗ {key}: DIFFERENT")


if __name__ == "__main__":
    test_review_time_decay()
    test_deterministic_with_decay()
