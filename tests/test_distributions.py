"""
Tests for the gamma distribution sampling in the SPL onsite generator.

This module tests the statistical properties and edge cases of the gamma
distribution implementation used for task creation and review times.
"""

import statistics
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_simulator import sample_gamma_from_mean_cv


class TestGammaDistribution:
    """Test gamma distribution sampling functionality."""

    def test_gamma_basic_sampling(self):
        """Test basic gamma distribution sampling with valid parameters."""
        # Test with reasonable parameters
        mean = 4.0
        cv = 0.25  # coefficient of variation (std/mean)
        samples = [sample_gamma_from_mean_cv(mean, cv) for _ in range(1000)]

        # All samples should be valid numbers
        assert all(isinstance(s, (int, float)) for s in samples)
        assert all(
            s > 0 for s in samples
        )  # Gamma distribution produces positive values

        # Statistical properties should be reasonable
        sample_mean = statistics.mean(samples)
        sample_std = statistics.stdev(samples)
        sample_cv = sample_std / sample_mean

        # Should be close to target (within 20% for large sample)
        assert abs(sample_mean - mean) / mean < 0.2
        assert abs(sample_cv - cv) / cv < 0.3  # More tolerance for CV

    def test_gamma_edge_cases(self):
        """Test gamma distribution with edge case parameters."""
        # Test with very small CV (low variability)
        samples = [sample_gamma_from_mean_cv(5.0, 0.1) for _ in range(100)]
        sample_std = statistics.stdev(samples)
        sample_mean = statistics.mean(samples)
        actual_cv = sample_std / sample_mean
        assert actual_cv < 0.2  # Should be small

        # Test with zero mean (should return fallback)
        sample = sample_gamma_from_mean_cv(0.0, 0.3)
        assert sample > 0

        # Test with zero CV (should return fallback)
        sample = sample_gamma_from_mean_cv(5.0, 0.0)
        assert sample > 0

    def test_gamma_negative_parameters(self):
        """Test gamma distribution handling of negative parameters."""
        # Should handle negative mean gracefully (fallback behavior)
        sample = sample_gamma_from_mean_cv(-1.0, 0.3)
        assert sample > 0

        # Should handle negative CV gracefully
        sample = sample_gamma_from_mean_cv(3.0, -0.5)
        assert sample > 0

    def test_gamma_consistency_with_seed(self):
        """Test that gamma sampling is consistent with same random seed."""
        # Set same seed for both sequences
        np.random.seed(42)
        samples1 = [sample_gamma_from_mean_cv(4.0, 0.3) for _ in range(20)]

        np.random.seed(42)
        samples2 = [sample_gamma_from_mean_cv(4.0, 0.3) for _ in range(20)]

        # Should produce identical sequences
        assert samples1 == samples2

    def test_gamma_statistical_properties(self):
        """Test that gamma distribution has expected statistical properties."""
        np.random.seed(12345)  # Fixed seed for reproducibility

        mean = 6.0
        cv = 0.4
        samples = [sample_gamma_from_mean_cv(mean, cv) for _ in range(2000)]

        sample_mean = statistics.mean(samples)
        sample_std = statistics.stdev(samples)
        sample_cv = sample_std / sample_mean

        # Check mean is close (within 10% for large sample)
        assert abs(sample_mean - mean) / mean < 0.1

        # Check coefficient of variation is reasonable
        assert abs(sample_cv - cv) / cv < 0.15

        # Gamma distribution should be right-skewed (mean > median)
        sample_median = statistics.median(samples)
        assert sample_mean > sample_median

    def test_gamma_different_parameters(self):
        """Test gamma distribution with various parameter combinations."""
        test_cases = [
            (1.0, 0.2),  # Low mean, low CV
            (10.0, 0.1),  # High mean, low CV
            (2.0, 0.8),  # Medium mean, high CV
            (50.0, 0.5),  # High mean, medium CV
        ]

        for mean, cv in test_cases:
            np.random.seed(42)  # Consistent seed for each test
            samples = [sample_gamma_from_mean_cv(mean, cv) for _ in range(500)]

            # All samples should be positive
            assert all(s > 0 for s in samples)

            # Mean should be approximately correct (within 15%)
            actual_mean = statistics.mean(samples)
            assert abs(actual_mean - mean) / mean < 0.15
