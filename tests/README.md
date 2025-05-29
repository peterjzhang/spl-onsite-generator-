# SPL Onsite Generator Test Suite

This directory contains the pytest-based test suite for the SPL onsite generator project. The tests are organized to provide comprehensive coverage of the simulation functionality.

## Test Structure

### Test Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_distributions.py`** - Tests for gamma distribution sampling (6 tests)
- **`test_agents.py`** - Tests for TrainerAgent and ReviewerAgent behavior (11 tests)  
- **`test_simulation.py`** - Tests for core simulation functionality (15 tests)

### Test Categories

#### Distribution Tests (`test_distributions.py`)
- Basic gamma distribution sampling
- Edge cases (zero/negative parameters)
- Statistical property validation
- Consistency with random seeding
- Various parameter combinations

#### Agent Tests (`test_agents.py`)
- **TrainerAgent**: Initialization, time tracking, task creation, writing, revision
- **ReviewerAgent**: Initialization, time tracking, reviewing, quality thresholds, time decay
- **Agent Interactions**: Cross-domain workflows, complete task lifecycles

#### Simulation Tests (`test_simulation.py`)
- **Determinism**: Same seeds produce identical results, different seeds differ
- **Basic Functionality**: Single/multi-domain, minimal configurations
- **Data Consistency**: Non-decreasing cumulative metrics, non-negative values
- **Edge Cases**: Zero-day, single-day, no trainers/reviewers, long simulations

## Running Tests

### Prerequisites
```bash
# Install dependencies (including pytest)
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_distributions.py

# Run specific test class
pytest tests/test_agents.py::TestTrainerAgent

# Run specific test method
pytest tests/test_simulation.py::TestSimulationDeterminism::test_deterministic_with_same_seed
```

### Advanced Options

```bash
# Run tests and show local variables on failure
pytest -v --tb=long

# Run only fast tests (exclude slow ones)
pytest -m "not slow"

# Run with coverage report
pytest --cov=task_simulator --cov-report=html

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Test Fixtures

The `conftest.py` file provides shared fixtures used across test files:

- **`basic_trainer_config`** - Standard trainer configuration
- **`basic_reviewer_config`** - Standard reviewer configuration  
- **`basic_domain_setup`** - Single domain with 2 trainers, 2 reviewers
- **`basic_simulation_config`** - 10-day simulation with basic domain
- **`seed_random`** - Function to seed random generators for deterministic tests

## Utility Functions

Helper functions for creating test configurations:

- `create_trainer_config(**overrides)` - Create trainer config with custom parameters
- `create_reviewer_config(**overrides)` - Create reviewer config with custom parameters
- `create_domain_setup(name, trainers, reviewers, ...)` - Create domain configuration
- `create_simulation_config(days, domains, seed)` - Create simulation configuration

## Test Coverage

Current test coverage includes:

- **32 total tests** across all modules
- **Distribution sampling**: 6 tests covering statistical accuracy and edge cases
- **Agent behavior**: 11 tests covering initialization, work processes, time tracking
- **Simulation core**: 15 tests covering determinism, consistency, edge cases

### Key Testing Areas

1. **Statistical Correctness**: Gamma distribution produces expected mean/variance
2. **Deterministic Behavior**: Same seeds always produce identical results
3. **Data Integrity**: Cumulative metrics never decrease, all values non-negative
4. **Edge Case Handling**: Zero agents, zero days, extreme configurations
5. **Agent Workflows**: Complete task lifecycle from creation to sign-off
6. **Time Decay**: Review time decreases with reviewer familiarity

## Adding New Tests

When adding new tests, follow these conventions:

1. **File Naming**: Use `test_*.py` pattern
2. **Class Naming**: Use `Test*` pattern (e.g., `TestNewFeature`)
3. **Method Naming**: Use `test_*` pattern (e.g., `test_new_functionality`)
4. **Use Fixtures**: Leverage shared fixtures from `conftest.py`
5. **Seed Random**: Use `seed_random()` fixture for deterministic tests
6. **Clear Assertions**: Use descriptive assertion messages

### Example Test

```python
def test_new_feature(self, seed_random):
    """Test description of what this test validates."""
    seed_random(42)
    
    # Setup
    config = create_simulation_config(simulation_days=5, random_seed=42)
    sim = Simulation(config)
    
    # Action
    results = sim.run()
    
    # Assertions
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 5
    assert results['some_metric'].sum() > 0
```

## Performance Testing

For performance-sensitive tests, use the `@pytest.mark.slow` decorator:

```python
@pytest.mark.slow
def test_long_simulation():
    """This test takes a long time to run."""
    config = create_simulation_config(simulation_days=365)
    # ... test implementation
```

Run fast tests only: `pytest -m "not slow"`

## Continuous Integration

The test suite is designed to run reliably in CI environments:

- All tests use fixed random seeds for deterministic results
- Tests complete in reasonable time (< 30 seconds total)
- No external dependencies beyond Python packages
- Clear failure messages for debugging

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Random Failures**: Check that tests use `seed_random()` fixture properly
3. **Slow Tests**: Use `pytest -x` to stop on first failure for faster debugging 