# Task Creation System Simulator Engine (`task_simulator.py`)

## 1. Overview

This simulation engine models a task creation and review workflow. It simulates a number of "Trainers" who create and revise tasks, and "Reviewers" who review these tasks, either signing them off or sending them back for revision. The simulation runs for a configurable number of days, tracking various metrics about task progression, agent workload, and task quality.

The engine is designed to be flexible, allowing for different configurations of agent behaviors, capacities, and task characteristics, potentially across multiple distinct "domains."

## 2. Core Components

The simulation is built around several key classes and concepts:

### 2.1. `TaskStatus` (Enum)
Defines the various states a task can be in throughout its lifecycle:
- `CLAIMED`: Task created, not yet worked on by a trainer.
- `WRITING_IN_PROGRESS`: Trainer is actively working on the initial creation.
- `COMPLETE`: Initial writing is done, task is ready for its first review.
- `REVIEW_IN_PROGRESS`: Reviewer is actively reviewing the task.
- `NEEDS_WORK`: Reviewer has reviewed the task and found issues requiring revision.
- `REVISION_IN_PROGRESS`: Trainer is actively working on revising the task.
- `FIXING_DONE`: Revision is complete, task is ready for subsequent review.
- `SIGNED_OFF`: Task has been reviewed and approved.

*(Note: `TaskStatus.REVIEW` is deprecated in favor of `REVIEW_IN_PROGRESS`)*

### 2.2. Configuration Classes
These dataclasses define the parameters for the simulation and its agents:

-   **`TrainerConfig`**: Settings for Trainer agents, including:
    -   Work capacity: `max_hours_per_week`, `target_hours_per_day` (and associated noise).
    -   Task effort: `writing_hours`, `revision_hours` (and noise).
    -   Quality parameters: `average_initial_quality` (and noise), `revision_improvement` (and noise).
    -   Behavior: `revision_priority` (likelihood of prioritizing revisions).
-   **`ReviewerConfig`**: Settings for Reviewer agents, including:
    -   Work capacity: `max_hours_per_week`, `target_hours_per_day` (and noise).
    -   Task effort: `review_hours` (and noise).
    -   Quality standard: `quality_threshold` for sign-off (and noise).
-   **`DomainSimulationSetup`**: Defines a specific "domain" within the simulation. Each domain has:
    -   `domain_name`: A unique name.
    -   `num_trainers`, `num_reviewers`: Number of each agent type in this domain.
    -   `trainer_cfg`, `reviewer_cfg`: Specific `TrainerConfig` and `ReviewerConfig` instances for agents in this domain.
-   **`SimulationConfig`**: Overall configuration for the simulation run:
    -   `simulation_days`: Total duration of the simulation.
    -   `week_length_days`: Defines a work week for resetting weekly hour limits.
    -   `domain_setups`: A list of `DomainSimulationSetup` objects, allowing for one or more domains to be simulated concurrently.

### 2.3. `Task`
Represents a single unit of work. Key attributes include:
-   `id`: Unique identifier.
-   `owner_id`, `owner_domain`: ID and domain of the trainer who created it.
-   `reviewer_id`, `reviewer_domain`: ID and domain of the reviewer who handled/is handling it.
-   `status`: Current `TaskStatus`.
-   `revision_count`: Number of times it has been revised.
-   `quality_score`: A numerical representation of its quality (0.0 to 1.0).
-   `minor_issues`, `major_issues`: Derived from `quality_score`.
-   Progress trackers: `writing_progress_hours`, `revision_progress_hours`, `review_progress_hours`.

### 2.4. Agent Classes

-   **`BaseAgent`**: The foundation for trainers and reviewers. Manages:
    -   `id`, `domain_name`.
    -   `cfg`: The specific `TrainerConfig` or `ReviewerConfig`.
    -   `actual_target_hours_per_day`: Daily hours sampled based on config + noise.
    -   `hours_worked_this_week`, `hours_worked_today`.
    -   `current_task_id`, `current_phase`: For tracking and prioritizing ongoing work.
    -   Methods for resetting hours and calculating available work increments.
-   **`TrainerAgent(BaseAgent)`**:
    -   Creates new tasks (`create_task`, then `work_on_task`).
    -   Revises tasks sent back by reviewers (`work_on_task`).
    -   Actual work times and quality outputs are sampled based on their `TrainerConfig` and associated noise parameters (e.g., `actual_writing_hours`, `actual_average_initial_quality`).
-   **`ReviewerAgent(BaseAgent)`**:
    -   Reviews tasks that are `COMPLETE` or `FIXING_DONE` (`work_on_review`).
    -   Decides to `SIGNED_OFF` or `NEEDS_WORK` based on the task's `quality_score` and their `actual_quality_threshold`.
    -   Actual review times and quality thresholds are sampled based on their `ReviewerConfig` and noise.

### 2.5. `Simulation`
The main orchestrator of the simulation.
-   Initializes agents based on `SimulationConfig` and its `domain_setups`.
-   Manages the list of all `tasks`.
-   Iterates through each `day` of the simulation.
-   Handles daily and weekly hour resets for agents.
-   Manages the agent work loop within each day.
-   Collects and returns a daily summary of metrics.

## 3. Simulation Logic

### 3.1. Daily Cycle
For each day in the simulation:
1.  **Weekly Reset**: If it's the start of a new week, all agents' `hours_worked_this_week` are reset.
2.  **Daily Reset**: All agents' `hours_worked_today` are reset.
3.  **Agent Shuffling**: The lists of trainers and reviewers are shuffled to introduce randomness in the order they act each day.
4.  **Agent Work Loop**: See below.
5.  **Metric Collection**: Key metrics for the day are collected.

### 3.2. Agent Work Loop
Within each day, the simulation enters a loop that continues as long as any agent performs work in a given pass. Work is processed in discrete `WORK_INCREMENT_HOURS` (default 0.5 hours).
-   For each trainer and then for each reviewer:
    -   The agent's `get_available_time_increment()` is checked. If zero, the agent does nothing.
    -   The simulation calls helper methods (`_process_trainer_actions` or `_process_reviewer_actions`) to determine agent actions.

### 3.3. Trainer Logic (`_process_trainer_actions`)
A trainer, if they have available time, will take one primary action per work increment opportunity:
1.  **Continue Current Task**: If `current_task_id` is set (meaning they were part-way through writing or revising), they continue working on that task. Progress is updated, and if the required hours are met, the task's status changes (e.g., to `COMPLETE` or `FIXING_DONE`) and quality is set/updated.
2.  **Start Revision**: If not continuing a task, and their `revision_priority` (a random chance) is met, they will look for one of their owned tasks that is in `NEEDS_WORK` or `REVISION_IN_PROGRESS` (not current). They prioritize tasks already `REVISION_IN_PROGRESS`. If found, they start/continue working on it.
3.  **Start/Continue Writing New Task**: If neither of the above, they look for one of their owned tasks that is `CLAIMED` or `WRITING_IN_PROGRESS` (not current). If found, they start/continue.
4.  **Create New Task**: If none of the above, they will `create_task()`. This new task is added to the simulation, and the trainer attempts to immediately start `work_on_task` for its initial writing phase in the same increment if time allows.

### 3.4. Reviewer Logic (`_process_reviewer_actions`)
A reviewer, if they have available time, will take one primary action per work increment opportunity:
1.  **Continue Current Review**: If `current_task_id` is set (meaning they were part-way through a review), they continue `work_on_review`. If the review completes, the task status becomes `SIGNED_OFF` (if `quality_score` >= `actual_quality_threshold`) or `NEEDS_WORK`. The task's `reviewer_id` and `reviewer_domain` are set.
2.  **Pick New Task for Review**: If not continuing a review, they search for an available task.
    -   They prioritize tasks already in `REVIEW_IN_PROGRESS` assigned to them (e.g., if the simulation was reloaded).
    -   Then, they look for tasks in `COMPLETE` or `FIXING_DONE` status.
    -   **Domain Restriction**: A reviewer will only pick up new/revised tasks where the `task.owner_domain` matches their own `reviewer.domain_name`.
    -   Tasks are sorted to prioritize `COMPLETE` (new) tasks over `FIXING_DONE` (revised) tasks.
    -   If a suitable task is found, they start `work_on_review`.

### 3.5. Multi-Domain Functionality
-   The `SimulationConfig` accepts a list of `DomainSimulationSetup` objects.
-   Each `DomainSimulationSetup` defines the number of trainers/reviewers and their specific `TrainerConfig` and `ReviewerConfig` for a named domain.
-   Agents are initialized with a `domain_name`.
-   Tasks store `owner_domain` (from the trainer) and `reviewer_domain` (from the reviewer who claims it).
-   As noted above, reviewers are restricted to picking up new tasks from their own domain.

## 4. Key Parameters & Noise Injection

-   Many aspects of agent behavior are configurable via `TrainerConfig` and `ReviewerConfig` (e.g., hours per task, quality levels, thresholds).
-   **Noise**: Most base parameters in the configs have an associated `_noise` parameter (e.g., `writing_hours_noise`).
-   During agent initialization (`__post_init__`), "actual" parameters (e.g., `actual_writing_hours`, `actual_average_initial_quality`, `actual_target_hours_per_day`) are sampled from a normal distribution where the mean is the base config value and the standard deviation is the `_noise` value.
-   Values are clamped/clipped to sensible ranges (e.g., quality between 0.0 and 1.0, hours >= 0.1).
-   This allows for heterogeneity among agents even if they share the same base configuration template.

## 5. Output

### 5.1. Daily Summary DataFrame
The `Simulation.run()` method returns a `pandas.DataFrame`. Each row represents one day of the simulation and contains metrics such as:
-   `day`: The simulation day number.
-   `new_tasks_created`: Number of tasks that entered `CLAIMED` status.
-   `tasks_writing_completed`: Tasks that moved to `COMPLETE`.
-   `tasks_revision_completed`: Tasks that moved to `FIXING_DONE`.
-   `tasks_decisioned_by_review`: Tasks that were either `SIGNED_OFF` or marked `NEEDS_WORK`.
-   End-of-day counts for tasks in various statuses (e.g., `tasks_claimed_eod`, `tasks_needing_work_eod`).
-   `signed_off_cumulative`: Cumulative count of signed-off tasks.
-   `avg_quality_signed_off`: Average quality of all tasks signed off so far.
-   Average hours worked by trainers/reviewers today.
-   `total_tasks_in_system`: Total number of tasks (active or signed-off).

### 5.2. Final Task States
After the simulation runs, the `simulation.tasks` attribute (a list of `Task` objects) contains the final state of every task created during the simulation. This can be used for more detailed post-simulation analysis.

### 5.3. Agent States
The `simulation.trainers` and `simulation.reviewers` lists contain the final states of all agent objects, which include their actual sampled parameters and total work metrics.

## 6. How to Use (Basic Example)

```python
from task_simulator import (
    TrainerConfig, ReviewerConfig, 
    DomainSimulationSetup, SimulationConfig, Simulation
)

# 1. Define agent configurations
trainer_config_eng = TrainerConfig(
    writing_hours=8, average_initial_quality=0.6, revision_priority=0.8
)
reviewer_config_eng = ReviewerConfig(
    review_hours=3, quality_threshold=0.85
)

# 2. Define domain setup(s)
eng_domain = DomainSimulationSetup(
    domain_name="Engineering",
    num_trainers=5,
    num_reviewers=2,
    trainer_cfg=trainer_config_eng,
    reviewer_cfg=reviewer_config_eng
)

# 3. Define overall simulation configuration
sim_config = SimulationConfig(
    simulation_days=60, # Simulate for 60 days
    domain_setups=[eng_domain]
)

# 4. Create and run the simulation
simulation = Simulation(config=sim_config)
daily_summary_df = simulation.run()

# 5. Access results
print("Daily Summary:")
print(daily_summary_df.tail())

print(f"\nTotal tasks at end: {len(simulation.tasks)}")
# Further analysis can be done on daily_summary_df and simulation.tasks
```

## 7. File Structure
The core simulation engine logic is self-contained within `task_simulator.py`.

## 8. Potential Extensions
-   Cost analysis based on hourly rates per domain.
-   More sophisticated task routing or reviewer assignment logic.
-   Agent learning or skill progression over time.
-   Different distributions for parameter noise.
-   Dependencies between tasks.
