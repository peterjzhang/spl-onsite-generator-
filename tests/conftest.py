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
    TrainerConfig,
    ReviewerConfig,
)


@pytest.fixture
def basic_trainer_config():
    """Basic trainer configuration for testing."""
    return TrainerConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=6.0,
        target_hours_per_day_noise=0.0,
        writing_hours=4.0,
        writing_hours_noise=0.0,
        revision_hours=2.0,
        revision_hours_noise=0.0,
        average_initial_quality=0.7,
        average_initial_quality_noise=0.0,
        revision_improvement=0.1,
        revision_improvement_noise=0.0,
        revision_priority=0.7,
    )


@pytest.fixture
def basic_reviewer_config():
    """Basic reviewer configuration for testing."""
    return ReviewerConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=5.0,
        target_hours_per_day_noise=0.0,
        review_hours=3.0,
        review_hours_noise=0.0,
        review_time_decay=0.8,
        quality_threshold=0.8,
        quality_threshold_noise=0.0,
    )


@pytest.fixture
def basic_domain_setup(basic_trainer_config, basic_reviewer_config):
    """Basic domain setup for testing."""
    return DomainSimulationSetup(
        domain_name="test_domain",
        num_trainers=2,
        num_reviewers=2,
        trainer_cfg=basic_trainer_config,
        reviewer_cfg=basic_reviewer_config,
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
def create_trainer_config():
    """Create a trainer configuration with optional overrides."""

    def _create(**overrides) -> TrainerConfig:
        defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 6.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 4.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 2.0,
            "revision_hours_noise": 0.0,
            "average_initial_quality": 0.7,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.1,
            "revision_improvement_noise": 0.0,
            "revision_priority": 0.7,
        }
        defaults.update(overrides)
        return TrainerConfig(**defaults)

    return _create


@pytest.fixture
def create_reviewer_config():
    """Create a reviewer configuration with optional overrides."""

    def _create(**overrides) -> ReviewerConfig:
        defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 5.0,
            "target_hours_per_day_noise": 0.0,
            "review_hours": 3.0,
            "review_hours_noise": 0.0,
            "review_time_decay": 0.8,
            "quality_threshold": 0.8,
            "quality_threshold_noise": 0.0,
        }
        defaults.update(overrides)
        return ReviewerConfig(**defaults)

    return _create


@pytest.fixture
def create_domain_setup():
    """Create a domain setup with optional overrides."""

    def _create(
        domain_name="test",
        num_trainers=2,
        num_reviewers=2,
        trainer_overrides=None,
        reviewer_overrides=None,
    ) -> DomainSimulationSetup:
        trainer_defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 6.0,
            "target_hours_per_day_noise": 0.0,
            "writing_hours": 4.0,
            "writing_hours_noise": 0.0,
            "revision_hours": 2.0,
            "revision_hours_noise": 0.0,
            "average_initial_quality": 0.7,
            "average_initial_quality_noise": 0.0,
            "revision_improvement": 0.1,
            "revision_improvement_noise": 0.0,
            "revision_priority": 0.7,
        }
        trainer_defaults.update(trainer_overrides or {})
        trainer_cfg = TrainerConfig(**trainer_defaults)

        reviewer_defaults = {
            "max_hours_per_week": 40.0,
            "target_hours_per_day": 5.0,
            "target_hours_per_day_noise": 0.0,
            "review_hours": 3.0,
            "review_hours_noise": 0.0,
            "review_time_decay": 0.8,
            "quality_threshold": 0.8,
            "quality_threshold_noise": 0.0,
        }
        reviewer_defaults.update(reviewer_overrides or {})
        reviewer_cfg = ReviewerConfig(**reviewer_defaults)

        return DomainSimulationSetup(
            domain_name=domain_name,
            num_trainers=num_trainers,
            num_reviewers=num_reviewers,
            trainer_cfg=trainer_cfg,
            reviewer_cfg=reviewer_cfg,
        )

    return _create


@pytest.fixture
def create_simulation_config():
    """Create a simulation configuration with optional domain overrides."""

    def _create(simulation_days=10, domains=None, random_seed=42) -> SimulationConfig:
        if domains is None:
            # Create default domain
            trainer_cfg = TrainerConfig(
                max_hours_per_week=40.0,
                target_hours_per_day=6.0,
                target_hours_per_day_noise=0.0,
                writing_hours=4.0,
                writing_hours_noise=0.0,
                revision_hours=2.0,
                revision_hours_noise=0.0,
                average_initial_quality=0.7,
                average_initial_quality_noise=0.0,
                revision_improvement=0.1,
                revision_improvement_noise=0.0,
                revision_priority=0.7,
            )
            reviewer_cfg = ReviewerConfig(
                max_hours_per_week=40.0,
                target_hours_per_day=5.0,
                target_hours_per_day_noise=0.0,
                review_hours=3.0,
                review_hours_noise=0.0,
                review_time_decay=0.8,
                quality_threshold=0.8,
                quality_threshold_noise=0.0,
            )
            domains = [
                DomainSimulationSetup(
                    domain_name="test",
                    num_trainers=2,
                    num_reviewers=2,
                    trainer_cfg=trainer_cfg,
                    reviewer_cfg=reviewer_cfg,
                )
            ]

        return SimulationConfig(
            simulation_days=simulation_days,
            domain_setups=domains,
            random_seed=random_seed,
        )

    return _create
