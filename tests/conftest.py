"""
Pytest configuration and shared fixtures for SPL onsite generator tests.

This file provides common fixtures and utilities used across all test modules.
"""

import pytest
import random
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_simulator import (
    SimulationConfig,
    DomainSimulationSetup,
    PerformanceLevelConfig,
)


@pytest.fixture
def basic_top_performer_config():
    """Basic top performer configuration for testing."""
    return PerformanceLevelConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=8.0,
        target_hours_per_day_noise=0.0,
        writing_hours=4.0,
        writing_hours_noise=0.0,
        revision_hours=1.0,
        revision_hours_noise=0.0,
        review_hours=2.0,
        review_hours_noise=0.0,
        average_initial_quality=0.8,
        average_initial_quality_noise=0.0,
        revision_improvement=0.1,
        revision_improvement_noise=0.0,
        quality_threshold=0.85,
        quality_threshold_noise=0.0,
        review_time_decay=0.9,
        revision_priority=0.7,
        review_time_percentage=0.5,
    )


@pytest.fixture
def basic_normal_contractor_config():
    """Basic normal contractor configuration for testing."""
    return PerformanceLevelConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=8.0,
        target_hours_per_day_noise=0.0,
        writing_hours=6.0,
        writing_hours_noise=0.0,
        revision_hours=2.0,
        revision_hours_noise=0.0,
        review_hours=3.0,
        review_hours_noise=0.0,
        average_initial_quality=0.7,
        average_initial_quality_noise=0.0,
        revision_improvement=0.08,
        revision_improvement_noise=0.0,
        quality_threshold=0.75,
        quality_threshold_noise=0.0,
        review_time_decay=0.95,
        revision_priority=0.6,
        review_time_percentage=0.0,
    )


@pytest.fixture
def basic_bad_contractor_config():
    """Basic bad contractor configuration for testing."""
    return PerformanceLevelConfig(
        max_hours_per_week=35.0,
        target_hours_per_day=7.0,
        target_hours_per_day_noise=0.0,
        writing_hours=10.0,
        writing_hours_noise=0.0,
        revision_hours=4.0,
        revision_hours_noise=0.0,
        review_hours=5.0,
        review_hours_noise=0.0,
        average_initial_quality=0.5,
        average_initial_quality_noise=0.0,
        revision_improvement=0.05,
        revision_improvement_noise=0.0,
        quality_threshold=0.65,
        quality_threshold_noise=0.0,
        review_time_decay=1.0,
        revision_priority=0.5,
        review_time_percentage=0.0,
    )


@pytest.fixture
def basic_domain_setup(
    basic_top_performer_config,
    basic_normal_contractor_config,
    basic_bad_contractor_config,
):
    """Basic domain setup for testing."""
    return DomainSimulationSetup(
        domain_name="test_domain",
        num_top_performers=1,
        num_normal_contractors=1,
        num_bad_contractors=1,
        top_performer_cfg=basic_top_performer_config,
        normal_contractor_cfg=basic_normal_contractor_config,
        bad_contractor_cfg=basic_bad_contractor_config,
    )


@pytest.fixture
def basic_simulation_config(basic_domain_setup):
    """Basic simulation configuration for testing."""
    return SimulationConfig(
        simulation_days=10, domain_setups=[basic_domain_setup], random_seed=42
    )


@pytest.fixture
def seed_random():
    """Fixture to seed random generators for deterministic testing."""

    def _seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    return _seed


@pytest.fixture
def create_performance_config():
    """Create a performance level configuration with optional overrides."""

    def _create(**overrides) -> PerformanceLevelConfig:
        defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 8.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 5.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 2.0,
            "revision_hours_noise": 0.0,
            "review_hours": 2.5,
            "review_hours_noise": 0.0,
            "average_initial_quality": 0.7,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.1,
            "revision_improvement_noise": 0.0,
            "quality_threshold": 0.8,
            "quality_threshold_noise": 0.0,
            "review_time_decay": 0.9,
            "revision_priority": 0.7,
            "review_time_percentage": 0.3,
        }
        defaults.update(overrides)
        return PerformanceLevelConfig(**defaults)

    return _create


@pytest.fixture
def create_domain_setup():
    """Create a domain setup with optional overrides."""

    def _create(
        domain_name="test",
        num_top_performers=1,
        num_normal_contractors=1,
        num_bad_contractors=0,
        top_performer_overrides=None,
        normal_contractor_overrides=None,
        bad_contractor_overrides=None,
    ) -> DomainSimulationSetup:

        # Top performer defaults
        top_performer_defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 8.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 4.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 1.0,
            "revision_hours_noise": 0.0,
            "review_hours": 2.0,
            "review_hours_noise": 0.0,
            "average_initial_quality": 0.8,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.1,
            "revision_improvement_noise": 0.0,
            "quality_threshold": 0.85,
            "quality_threshold_noise": 0.0,
            "review_time_decay": 0.9,
            "revision_priority": 0.7,
            "review_time_percentage": 0.5,
        }
        top_performer_defaults.update(top_performer_overrides or {})
        top_performer_cfg = PerformanceLevelConfig(**top_performer_defaults)

        # Normal contractor defaults
        normal_contractor_defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 8.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 6.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 2.0,
            "revision_hours_noise": 0.0,
            "review_hours": 3.0,
            "review_hours_noise": 0.0,
            "average_initial_quality": 0.7,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.08,
            "revision_improvement_noise": 0.0,
            "quality_threshold": 0.75,
            "quality_threshold_noise": 0.0,
            "review_time_decay": 0.95,
            "revision_priority": 0.6,
            "review_time_percentage": 0.0,
        }
        normal_contractor_defaults.update(normal_contractor_overrides or {})
        normal_contractor_cfg = PerformanceLevelConfig(**normal_contractor_defaults)

        # Bad contractor defaults
        bad_contractor_defaults = {
            "max_hours_per_week": 35.0,
            "target_hours_per_day": 7.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 10.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 4.0,
            "revision_hours_noise": 0.0,
            "review_hours": 5.0,
            "review_hours_noise": 0.0,
            "average_initial_quality": 0.5,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.05,
            "revision_improvement_noise": 0.0,
            "quality_threshold": 0.65,
            "quality_threshold_noise": 0.0,
            "review_time_decay": 1.0,
            "revision_priority": 0.5,
            "review_time_percentage": 0.0,
        }
        bad_contractor_defaults.update(bad_contractor_overrides or {})
        bad_contractor_cfg = PerformanceLevelConfig(**bad_contractor_defaults)

        return DomainSimulationSetup(
            domain_name=domain_name,
            num_top_performers=num_top_performers,
            num_normal_contractors=num_normal_contractors,
            num_bad_contractors=num_bad_contractors,
            top_performer_cfg=top_performer_cfg,
            normal_contractor_cfg=normal_contractor_cfg,
            bad_contractor_cfg=bad_contractor_cfg,
        )

    return _create


@pytest.fixture
def create_simulation_config():
    """Create a simulation configuration with optional domain overrides."""

    def _create(simulation_days=10, domains=None, random_seed=42) -> SimulationConfig:
        if domains is None:
            # Create default domain with performance levels
            top_performer_cfg = PerformanceLevelConfig(
                max_hours_per_week=40.0,
                target_hours_per_day=8.0,
                target_hours_per_day_noise=0.0,
                writing_hours=4.0,
                writing_hours_noise=0.0,
                revision_hours=1.0,
                revision_hours_noise=0.0,
                review_hours=2.0,
                review_hours_noise=0.0,
                average_initial_quality=0.8,
                average_initial_quality_noise=0.0,
                revision_improvement=0.1,
                revision_improvement_noise=0.0,
                quality_threshold=0.85,
                quality_threshold_noise=0.0,
                review_time_decay=0.9,
                revision_priority=0.7,
                review_time_percentage=0.5,
            )
            normal_contractor_cfg = PerformanceLevelConfig(
                max_hours_per_week=40.0,
                target_hours_per_day=8.0,
                target_hours_per_day_noise=0.0,
                writing_hours=6.0,
                writing_hours_noise=0.0,
                revision_hours=2.0,
                revision_hours_noise=0.0,
                review_hours=3.0,
                review_hours_noise=0.0,
                average_initial_quality=0.7,
                average_initial_quality_noise=0.0,
                revision_improvement=0.08,
                revision_improvement_noise=0.0,
                quality_threshold=0.75,
                quality_threshold_noise=0.0,
                review_time_decay=0.95,
                revision_priority=0.6,
                review_time_percentage=0.0,
            )
            bad_contractor_cfg = PerformanceLevelConfig(
                max_hours_per_week=35.0,
                target_hours_per_day=7.0,
                target_hours_per_day_noise=0.0,
                writing_hours=10.0,
                writing_hours_noise=0.0,
                revision_hours=4.0,
                revision_hours_noise=0.0,
                review_hours=5.0,
                review_hours_noise=0.0,
                average_initial_quality=0.5,
                average_initial_quality_noise=0.0,
                revision_improvement=0.05,
                revision_improvement_noise=0.0,
                quality_threshold=0.65,
                quality_threshold_noise=0.0,
                review_time_decay=1.0,
                revision_priority=0.5,
                review_time_percentage=0.0,
            )
            domains = [
                DomainSimulationSetup(
                    domain_name="test",
                    num_top_performers=1,
                    num_normal_contractors=1,
                    num_bad_contractors=0,
                    top_performer_cfg=top_performer_cfg,
                    normal_contractor_cfg=normal_contractor_cfg,
                    bad_contractor_cfg=bad_contractor_cfg,
                )
            ]

        return SimulationConfig(
            simulation_days=simulation_days,
            domain_setups=domains,
            random_seed=random_seed,
        )

    return _create
